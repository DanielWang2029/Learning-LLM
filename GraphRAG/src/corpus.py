"""A tiny toy corpus of facts spanning four distinct themes.

Each fact is a (subject, relation, object, theme) tuple.  We render each fact as
a short natural sentence — those sentences are the "documents"/chunks a RAG
system would index.  A few *bridge* facts connect otherwise-separate themes, so
community detection has real work to do (the graph is a single connected
component).
"""

from __future__ import annotations

# Relation key -> the phrase used when rendering a sentence.
RELATIONS = {
    "founded": "founded",
    "develops": "develops",
    "located_in": "is located in",
    "works_at": "works at",
    "partners_with": "partners with",
    "invests_in": "invests in",
    "researches": "researches",
}

THEMES = ["Solar Energy", "Space Exploration", "Genomics", "Artificial Intelligence"]

# (subject, relation_key, object, theme)
FACTS = [
    # --- Solar Energy ---
    ("Aria", "founded", "Helios", "Solar Energy"),
    ("Helios", "develops", "SolarPanels", "Solar Energy"),
    ("Helios", "located_in", "Nevada", "Solar Energy"),
    ("Bruno", "works_at", "Helios", "Solar Energy"),
    ("Helios", "partners_with", "GridCo", "Solar Energy"),
    ("GridCo", "develops", "Batteries", "Solar Energy"),
    # --- Space Exploration ---
    ("Cara", "founded", "Orion", "Space Exploration"),
    ("Orion", "develops", "Rockets", "Space Exploration"),
    ("Orion", "located_in", "Texas", "Space Exploration"),
    ("Dan", "works_at", "Orion", "Space Exploration"),
    ("Orion", "partners_with", "StarPort", "Space Exploration"),
    ("StarPort", "develops", "Satellites", "Space Exploration"),
    # --- Genomics ---
    ("Ela", "founded", "Genoma", "Genomics"),
    ("Genoma", "develops", "GeneTherapy", "Genomics"),
    ("Genoma", "located_in", "Boston", "Genomics"),
    ("Finn", "works_at", "Genoma", "Genomics"),
    ("Genoma", "partners_with", "BioLab", "Genomics"),
    ("BioLab", "researches", "DNA", "Genomics"),
    # --- Artificial Intelligence ---
    ("Gia", "founded", "Cortex", "Artificial Intelligence"),
    ("Cortex", "develops", "NeuralNets", "Artificial Intelligence"),
    ("Cortex", "located_in", "Seattle", "Artificial Intelligence"),
    ("Hugo", "works_at", "Cortex", "Artificial Intelligence"),
    ("Cortex", "partners_with", "DataWorks", "Artificial Intelligence"),
    ("DataWorks", "develops", "Chips", "Artificial Intelligence"),
    # --- Bridge facts (connect themes into one graph) ---
    ("Nova", "invests_in", "Helios", "Solar Energy"),
    ("Nova", "invests_in", "Orion", "Space Exploration"),
    ("Quill", "invests_in", "Genoma", "Genomics"),
    ("Quill", "invests_in", "Cortex", "Artificial Intelligence"),
]


def render_sentence(fact) -> str:
    subj, rel, obj, _ = fact
    return f"{subj} {RELATIONS[rel]} {obj}."


def build_chunks():
    """Return the corpus as a list of (text, theme) chunks — the RAG documents."""
    return [(render_sentence(f), f[3]) for f in FACTS]
