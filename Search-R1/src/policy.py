"""The search policy (RL-trained) and a no-search baseline (Search-R1, 2025).

The policy learns *when and what to search*, purely from a correctness reward, by
REINFORCE with a group-relative baseline (GRPO-style, no value network). Its
action logits are a linear function of **type-based features** only — it never sees
entity identities — so the learned search *procedure* generalizes to entities it
was never trained on. That is Search-R1's whole point: retrieval-augmented
reasoning transfers, memorization does not.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .corpus import COUNTRY, TYPE_NAMES

N_TYPES = 5
FEAT_DIM = N_TYPES + 2  # type one-hot + [is_search, is_answer]


def _search_feat(entity_type: int) -> list[float]:
    f = [0.0] * FEAT_DIM
    f[entity_type] = 1.0
    f[N_TYPES] = 1.0  # is_search
    return f


def _answer_feat() -> list[float]:
    f = [0.0] * FEAT_DIM
    f[N_TYPES + 1] = 1.0  # is_answer
    return f


class SearchPolicy(nn.Module):
    """Scores each available action from type-based features (identity-agnostic)."""

    def __init__(self):
        super().__init__()
        self.score = nn.Linear(FEAT_DIM, 1)
        # Zero init -> all actions equally likely at first (uniform exploration),
        # and greedy decoding falls through to the first-listed action (answer),
        # so an untrained policy answers immediately and is wrong.
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def rollout(self, corpus, person_id, max_hops=4, sample=True, generator=None,
                cost_per_search=0.2):
        """Run one interleaved reason→search episode.

        Returns a dict with the final answer, a readable trace, the number of
        searches, whether it was correct, the reward, and the summed log-prob of
        the actions taken (differentiable, for the policy gradient).
        """
        known = [person_id]              # discovered entities (search candidates)
        searched = set()
        discovered_country = None
        trace = []
        logp = torch.zeros(())
        n_searches = 0

        for _ in range(max_hops + 1):
            # Build the available action set: answer with the best country found so
            # far, or search any un-searched known entity. Answer is listed first so
            # an *untrained* greedy policy answers prematurely (wrong) — RL must
            # learn to search before answering.
            actions = [("answer", discovered_country)]   # ("search", e)/("answer", c)
            feats = [_answer_feat()]
            for e in known:
                if e not in searched:
                    actions.append(("search", e))
                    feats.append(_search_feat(corpus.type_of[e]))

            logits = self.score(torch.tensor(feats)).squeeze(-1)
            probs = F.softmax(logits, dim=-1)
            if sample:
                idx = int(torch.multinomial(probs, 1, generator=generator))
            else:
                idx = int(torch.argmax(probs))
            logp = logp + torch.log(probs[idx] + 1e-9)

            kind, target = actions[idx]
            if kind == "answer":
                trace.append(("answer", target))
                final = target
                break
            # search: retrieve facts, reveal new entities
            searched.add(target)
            n_searches += 1
            results = corpus.search(target)
            trace.append(("search", target, results))
            for rel, tgt in results:
                if tgt not in known:
                    known.append(tgt)
                if corpus.type_of[tgt] == COUNTRY:
                    discovered_country = tgt
        else:
            final = discovered_country  # ran out of hops

        correct = (final is not None and final == corpus.answer_country(person_id))
        reward = (1.0 if correct else 0.0) - cost_per_search * n_searches
        return {"answer": final, "trace": trace, "n_searches": n_searches,
                "correct": correct, "reward": reward, "logp": logp}


class NoSearchClassifier(nn.Module):
    """Baseline: map a person id directly to a country, with NO retrieval.

    It can only memorize person→country for persons seen in training, so it fails
    on held-out persons — exactly the gap retrieval closes.
    """

    def __init__(self, n_persons: int, n_countries: int):
        super().__init__()
        self.emb = nn.Embedding(n_persons, 32)
        self.out = nn.Linear(32, n_countries)

    def forward(self, person_local_idx: torch.Tensor) -> torch.Tensor:
        return self.out(F.relu(self.emb(person_local_idx)))
