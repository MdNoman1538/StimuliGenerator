"""
Brysbaert, Warriner & Kuperman (2014) Concreteness Ratings
────────────────────────────────────────────────────────────
Concreteness ratings on a 1–5 scale (1=abstract, 5=concrete).

Loads the full 39,954-word dataset from:
  Concreteness_ratings_Brysbaert_et_al_BRM.txt  (tab-separated, same directory)

Falls back to a small hardcoded subset if the file is not found.

Reference:
Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014). Concreteness ratings for 40
thousand generally known English word lemmas. Behavior Research Methods, 46(3), 904–911.
"""

import os as _os
import csv as _csv

# ── hardcoded fallback (used only when the full data file is absent) ──────────
_FALLBACK = {
    "idea": 1.54, "concept": 1.54, "thought": 1.54, "belief": 1.58,
    "feeling": 1.76, "emotion": 1.83, "knowledge": 1.67, "understanding": 1.64,
    "theory": 1.69, "justice": 1.85, "love": 2.15, "hope": 1.83,
    "fear": 2.00, "truth": 1.92, "freedom": 2.08, "mind": 1.92,
    "system": 2.08, "structure": 2.38, "effort": 2.31, "support": 2.54,
    "process": 2.46, "method": 2.31, "design": 3.08, "object": 3.54,
    "material": 3.31, "tool": 4.92, "hand": 4.92, "glass": 4.92,
    "food": 4.77, "part": 3.62, "motion": 3.23, "hold": 3.54,
    "push": 3.54, "pull": 3.54, "grip": 4.52,
}


def _load_full_dataset(path: str) -> dict[str, float]:
    """Parse the tab-separated Brysbaert BRM file into {word: conc_mean}."""
    data: dict[str, float] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = _csv.DictReader(fh, delimiter="\t")
        for row in reader:
            word = row.get("Word", "").strip().lower()
            raw  = row.get("Conc.M", "").strip()
            if word and raw:
                try:
                    data[word] = float(raw)
                except ValueError:
                    pass
    return data


def _init() -> dict[str, float]:
    here = _os.path.dirname(_os.path.abspath(__file__))
    candidate = _os.path.join(here, "Concreteness_ratings_Brysbaert_et_al_BRM.txt")
    if _os.path.isfile(candidate):
        try:
            loaded = _load_full_dataset(candidate)
            if len(loaded) > 1000:          # sanity-check: real file has ~40k rows
                return loaded
        except Exception:
            pass
    return dict(_FALLBACK)


BRYSBAERT_CONCRETENESS: dict[str, float] = _init()


def get_concreteness(word: str) -> float | None:
    """Return concreteness score (1–5) for a word, or None if not found."""
    return BRYSBAERT_CONCRETENESS.get(word.lower().strip())


def normalize_concreteness(score: float,
                           min_val: float = 1.0,
                           max_val: float = 5.0) -> float:
    """Normalize concreteness (1–5) to abstraction level in [0, 1].

    1.0 (most abstract) → 1.0
    5.0 (most concrete) → 0.0
    """
    if score is None:
        return None
    return float(max(0.0, min(1.0, 1.0 - (score - min_val) / (max_val - min_val))))
