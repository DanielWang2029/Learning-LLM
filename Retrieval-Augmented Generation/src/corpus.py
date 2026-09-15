"""A tiny knowledge corpus of *invented* facts, plus questions over it.

The entities (Zubland, Marovia, ...) are fictional, so a closed-book model has
no way to know these facts from its parameters — it can only guess. That is
exactly the point: RAG grounds the answer in retrieved text, while a
parametric-only model hallucinates (paper Section 1, "hallucination" and
"knowledge-intensive tasks").

Every fact is written in a uniform template so the extractive reader can pull
the answer straight out of the retrieved sentence:

    "The {relation} of {entity} is {value}."
"""

from __future__ import annotations

# (entity, relation, value) triples about six fictional countries.
FACTS = [
    ("Zubland",  "capital",           "Marn"),
    ("Zubland",  "currency",          "zub"),
    ("Zubland",  "national animal",   "quokla"),
    ("Zubland",  "tallest mountain",  "Mount Ubbo"),
    ("Zubland",  "official language", "Zubbish"),
    ("Zubland",  "founder",           "Queen Ryla"),

    ("Marovia",  "capital",           "Deltra"),
    ("Marovia",  "currency",          "marov"),
    ("Marovia",  "national animal",   "fennter"),
    ("Marovia",  "tallest mountain",  "Mount Cael"),
    ("Marovia",  "official language", "Marovian"),
    ("Marovia",  "founder",           "Duke Osro"),

    ("Qentaria", "capital",           "Yselle"),
    ("Qentaria", "currency",          "qent"),
    ("Qentaria", "national animal",   "brindle"),
    ("Qentaria", "tallest mountain",  "Mount Torv"),
    ("Qentaria", "official language", "Qenti"),
    ("Qentaria", "founder",           "Elder Pahl"),

    ("Vondel",   "capital",           "Aikon"),
    ("Vondel",   "currency",          "vond"),
    ("Vondel",   "national animal",   "sprightle"),
    ("Vondel",   "tallest mountain",  "Mount Garn"),
    ("Vondel",   "official language", "Vondish"),
    ("Vondel",   "founder",           "Captain Idris"),

    ("Askoria",  "capital",           "Threa"),
    ("Askoria",  "currency",          "asko"),
    ("Askoria",  "national animal",   "mirral"),
    ("Askoria",  "tallest mountain",  "Mount Selk"),
    ("Askoria",  "official language", "Askorian"),
    ("Askoria",  "founder",           "Sister Neve"),

    ("Prenia",   "capital",           "Oloe"),
    ("Prenia",   "currency",          "pren"),
    ("Prenia",   "national animal",   "tovex"),
    ("Prenia",   "tallest mountain",  "Mount Pell"),
    ("Prenia",   "official language", "Prenic"),
    ("Prenia",   "founder",           "Baron Voss"),
]


def _doc(entity: str, relation: str, value: str) -> str:
    return f"The {relation} of {entity} is {value}."


# The corpus the retriever searches over (one sentence per fact).
DOCS = [_doc(e, r, v) for (e, r, v) in FACTS]


def build_questions():
    """Return a list of question dicts with the gold doc id and gold answer."""
    questions = []
    for i, (entity, relation, value) in enumerate(FACTS):
        questions.append({
            "question": f"What is the {relation} of {entity}?",
            "entity": entity,
            "relation": relation,
            "answer": value,
            "gold_doc_id": i,
        })
    return questions


# A small, fixed sample of questions to display in the demo (varied relations).
DEMO_QUESTION_IDS = [0, 7, 14, 21, 26, 33, 3, 11]


# What a *closed-book* (parametric-only) model "knows": real-world associations.
# For the fictional entities above these are all wrong — i.e. hallucinations.
CLOSED_BOOK_PRIORS = {
    "capital":           ["Paris", "London", "Tokyo", "Berlin", "Madrid", "Rome"],
    "currency":          ["dollar", "euro", "yen", "pound", "peso", "franc"],
    "national animal":   ["lion", "eagle", "tiger", "bear", "fox", "wolf"],
    "tallest mountain":  ["Mount Everest", "Mont Blanc", "Denali", "K2",
                          "Kilimanjaro", "Mount Fuji"],
    "official language": ["English", "French", "Spanish", "German",
                          "Mandarin", "Arabic"],
    "founder":           ["George Washington", "Julius Caesar", "Napoleon",
                          "Genghis Khan", "Cleopatra", "Augustus"],
}
