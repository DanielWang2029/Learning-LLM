"""Three agents on the same questions, to isolate what ReAct contributes.

* ``react_agent``      — interleaves Thought → Action → Observation, chaining
                          tool calls (the paper's method).
* ``reasoning_only``   — "chain of thought" with NO tools: it can reason but must
                          rely on flawed parametric memory, so it hallucinates.
* ``acting_only``      — calls tools but with NO reasoning to chain them: it fires
                          a single naive action and cannot do multi-hop.

Each returns a trace (list of steps) and a final answer, so we can print the
full interleaved reasoning+acting and compare correctness.
"""

from __future__ import annotations

from .environment import calc, lookup


# ---------------------------------------------------------------------------
# Ground truth (a reference solver over the KB) — used only to score answers.
# ---------------------------------------------------------------------------
def gold_answer(task: dict) -> str:
    k, p = task["kind"], task["params"]
    if k == "capital_of_residence":
        country = lookup(p["person"], "country").value
        return str(lookup(country, "capital").value)
    if k == "residence":
        return str(lookup(p["person"], "country").value)
    if k == "sum_population":
        a = lookup(lookup(p["p1"], "country").value, "population").value
        b = lookup(lookup(p["p2"], "country").value, "population").value
        return str(a + b)
    if k == "combined_attr":
        a = lookup(p["p1"], p["attr"]).value
        b = lookup(p["p2"], p["attr"]).value
        return str(a + b)
    if k == "wheels":
        w1 = lookup(p["thing1"], "wheels").value
        w2 = lookup(p["thing2"], "wheels").value
        return str(p["n1"] * w1 + p["n2"] * w2)
    raise ValueError(k)


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------
def _mk():
    steps: list[dict] = []

    def think(t):
        steps.append({"type": "thought", "text": t})

    def act(tool, args, res):
        steps.append({"type": "action", "tool": tool, "args": args,
                      "obs": res.text, "ok": res.ok})
        return res

    return steps, think, act


def _result(name, task, steps, answer):
    return {
        "name": name,
        "question": task["text"],
        "steps": steps + [{"type": "finish", "answer": str(answer)}],
        "answer": str(answer),
        "correct": str(answer) == gold_answer(task),
    }


# ---------------------------------------------------------------------------
# 1) ReAct — reason + act, chaining observations
# ---------------------------------------------------------------------------
def react_agent(task: dict) -> dict:
    k, p = task["kind"], task["params"]
    steps, think, act = _mk()
    ans = "unknown"

    if k == "capital_of_residence":
        person = p["person"]
        think(f"I need the capital of the country {person} lives in — first find {person}'s country.")
        r1 = act("Lookup", [person, "country"], lookup(person, "country"))
        country = r1.value
        think(f"{person} lives in {country}. Now look up the capital of {country}.")
        r2 = act("Lookup", [country, "capital"], lookup(country, "capital"))
        think(f"The capital of {country} is {r2.value}.")
        ans = r2.value

    elif k == "residence":
        person = p["person"]
        think(f"I just need the country {person} lives in.")
        r1 = act("Lookup", [person, "country"], lookup(person, "country"))
        ans = r1.value

    elif k == "sum_population":
        p1, p2 = p["p1"], p["p2"]
        think(f"I need the populations of the countries {p1} and {p2} live in, then add them.")
        c1 = act("Lookup", [p1, "country"], lookup(p1, "country")).value
        pop1 = act("Lookup", [c1, "population"], lookup(c1, "population")).value
        c2 = act("Lookup", [p2, "country"], lookup(p2, "country")).value
        pop2 = act("Lookup", [c2, "population"], lookup(c2, "population")).value
        think(f"{c1} has {pop1}M and {c2} has {pop2}M. Add them with the calculator.")
        ans = act("Calc", [f"{pop1}+{pop2}"], calc(f"{pop1}+{pop2}")).value

    elif k == "combined_attr":
        p1, p2, attr = p["p1"], p["p2"], p["attr"]
        think(f"I need {p1}'s and {p2}'s {attr}, then add them.")
        a = act("Lookup", [p1, attr], lookup(p1, attr)).value
        b = act("Lookup", [p2, attr], lookup(p2, attr)).value
        think(f"{p1}'s {attr} is {a}, {p2}'s is {b}. Sum them.")
        ans = act("Calc", [f"{a}+{b}"], calc(f"{a}+{b}")).value

    elif k == "wheels":
        n1, t1, n2, t2 = p["n1"], p["thing1"], p["n2"], p["thing2"]
        think(f"I need wheels-per-{t1} and wheels-per-{t2}, then compute the total.")
        w1 = act("Lookup", [t1, "wheels"], lookup(t1, "wheels")).value
        w2 = act("Lookup", [t2, "wheels"], lookup(t2, "wheels")).value
        expr = f"{n1}*{w1} + {n2}*{w2}"
        think(f"A {t1} has {w1} wheels, a {t2} has {w2}. Compute {expr}.")
        ans = act("Calc", [expr], calc(expr)).value

    return _result("ReAct (reason + act)", task, steps, ans)


# ---------------------------------------------------------------------------
# 2) Reasoning-only — no tools, flawed memory -> hallucinates
# ---------------------------------------------------------------------------
# A deliberately incomplete "parametric memory" (what the model thinks it knows).
_MEMORY_CAPITAL = "Paris"      # its single most-salient capital
_MEMORY_NUMBER = 100           # a round guess for any total it can't ground


def reasoning_only(task: dict) -> dict:
    k = task["kind"]
    steps, think, _ = _mk()
    if k in ("capital_of_residence",):
        think("I'll reason about capitals from memory (no tools). The most famous "
              f"capital I can recall is {_MEMORY_CAPITAL}, so I'll say that.")
        ans = _MEMORY_CAPITAL
    elif k == "residence":
        think("From memory I think this person lives in the United States.")
        ans = "United States"
    else:
        think("This needs numbers I don't have grounded access to; I'll estimate "
              f"the total as about {_MEMORY_NUMBER}.")
        ans = _MEMORY_NUMBER
    return _result("Reasoning-only (no tools)", task, steps, ans)


# ---------------------------------------------------------------------------
# 3) Acting-only — tools but no reasoning to chain -> one naive action
# ---------------------------------------------------------------------------
def acting_only(task: dict) -> dict:
    k, p = task["kind"], task["params"]
    steps, _, act = _mk()
    ans = "unknown"
    if k == "capital_of_residence":
        # Naively ask for the person's capital directly — no multi-hop.
        r = act("Lookup", [p["person"], "capital"], lookup(p["person"], "capital"))
        ans = r.value if r.ok else "unknown"
    elif k == "residence":
        r = act("Lookup", [p["person"], "country"], lookup(p["person"], "country"))
        ans = r.value if r.ok else "unknown"
    elif k == "sum_population":
        # No reasoning to go person -> country -> population, just try directly.
        r = act("Lookup", [p["p1"], "population"], lookup(p["p1"], "population"))
        ans = r.value if r.ok else "unknown"
    elif k == "combined_attr":
        # Fetches one value, returns it without adding the second (no plan).
        r = act("Lookup", [p["p1"], p["attr"]], lookup(p["p1"], p["attr"]))
        ans = r.value if r.ok else "unknown"
    elif k == "wheels":
        # Looks up one thing's wheels, returns it (ignores counts and the rest).
        r = act("Lookup", [p["thing1"], "wheels"], lookup(p["thing1"], "wheels"))
        ans = r.value if r.ok else "unknown"
    return _result("Acting-only (no reasoning)", task, steps, ans)


AGENTS = {
    "react": react_agent,
    "reasoning_only": reasoning_only,
    "acting_only": acting_only,
}
