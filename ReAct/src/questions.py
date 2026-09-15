"""The toy multi-hop question set (natural-language text + structured intent)."""

from __future__ import annotations

QUESTIONS: list[dict] = [
    {"text": "What is the capital of the country Alice lives in?",
     "kind": "capital_of_residence", "params": {"person": "Alice"}},
    {"text": "What is the capital of the country Bob lives in?",
     "kind": "capital_of_residence", "params": {"person": "Bob"}},
    {"text": "What is the total population (in millions) of the countries Alice and Carol live in?",
     "kind": "sum_population", "params": {"p1": "Alice", "p2": "Carol"}},
    {"text": "What is the combined age of Bob and Carol?",
     "kind": "combined_attr", "params": {"p1": "Bob", "p2": "Carol", "attr": "age"}},
    {"text": "How many wheels do 3 bicycles and 2 cars have in total?",
     "kind": "wheels", "params": {"n1": 3, "thing1": "bicycle", "n2": 2, "thing2": "car"}},
    {"text": "In which country does Carol live?",
     "kind": "residence", "params": {"person": "Carol"}},
]
