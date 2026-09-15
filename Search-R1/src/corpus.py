"""A tiny knowledge base + a real retrieval "search" tool (Search-R1, 2025).

Search-R1 trains a model to interleave reasoning with calls to a real search
engine. Here the "search engine" is a lookup over a small generated corpus of
facts. Answering a question requires *multi-hop* retrieval:

    "What country does Person_p live in?"
        hop 1: search Person_p     -> lives_in City_c   (+ distractor facts)
        hop 2: search City_c       -> located_in Country_k
        answer: Country_k

Only persons and cities have outgoing facts; countries, years and colors are
dead ends. The born_in/likes facts are distractors the policy must learn to skip.
"""

from __future__ import annotations

import torch

# Entity types.
PERSON, CITY, COUNTRY, YEAR, COLOR = 0, 1, 2, 3, 4
TYPE_NAMES = {PERSON: "Person", CITY: "City", COUNTRY: "Country",
              YEAR: "Year", COLOR: "Color"}


class Corpus:
    def __init__(self, n_persons=60, n_cities=15, n_countries=5, n_years=20,
                 n_colors=8, generator=None):
        g = generator or torch.Generator().manual_seed(0)
        self.type_of = {}          # entity id -> type
        self.name_of = {}          # entity id -> readable name
        self.facts = {}            # subject id -> list of (relation, target id)
        nid = 0

        def add(t, label):
            nonlocal nid
            eid = nid; nid += 1
            self.type_of[eid] = t
            self.name_of[eid] = f"{TYPE_NAMES[t]}_{label}"
            self.facts[eid] = []
            return eid

        self.countries = [add(COUNTRY, i) for i in range(n_countries)]
        self.years = [add(YEAR, 1970 + i) for i in range(n_years)]
        self.colors = [add(COLOR, i) for i in range(n_colors)]
        self.cities = []
        for i in range(n_cities):
            c = add(CITY, i)
            country = self.countries[int(torch.randint(n_countries, (1,), generator=g))]
            self.facts[c].append(("located_in", country))
            self.cities.append(c)
        self.persons = []
        for i in range(n_persons):
            p = add(PERSON, i)
            city = self.cities[int(torch.randint(n_cities, (1,), generator=g))]
            year = self.years[int(torch.randint(n_years, (1,), generator=g))]
            color = self.colors[int(torch.randint(n_colors, (1,), generator=g))]
            # lives_in is the useful fact; born_in / likes are distractors.
            self.facts[p].append(("lives_in", city))
            self.facts[p].append(("born_in", year))
            self.facts[p].append(("likes", color))
            self.persons.append(p)

    # -- the retrieval tool ---------------------------------------------------
    def search(self, entity_id: int):
        """Return the list of (relation, target_id) facts about ``entity_id``."""
        return list(self.facts.get(entity_id, []))

    def answer_country(self, person_id: int) -> int:
        """Ground-truth 2-hop answer: the country the person lives in."""
        city = self.facts[person_id][0][1]           # lives_in
        return self.facts[city][0][1]                # located_in

    def fact_str(self, subject_id: int) -> str:
        parts = [f"{self.name_of[subject_id]} {rel} {self.name_of[t]}"
                 for rel, t in self.facts[subject_id]]
        return "; ".join(parts) if parts else f"(no facts about {self.name_of[subject_id]})"
