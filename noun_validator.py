"""
Noun parity validator (Rule 8 enforcement).

Two strategies for slot extraction:

  1. extract_noun_slots(text)
     Uses spaCy noun_chunks. Position-based alignment between two texts —
     prone to cascading misalignment when chunk counts differ.

  2. extract_diff_slots(text_a, text_b)  ← used by check_noun_parity
     Auto-infers the locked skeleton by diffing the two texts. Words that
     match between A and B are the skeleton; gaps where they differ are the
     slots. Same number of slots in both, perfectly aligned, no cascade.
"""

import difflib
import re

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    _nlp = None
    SPACY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenization
# ─────────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b[\w'-]+\b|[^\w\s]")


def _tokenize(text: str) -> list[str]:
    """Word-level tokenization, keeping punctuation as separate tokens."""
    return _WORD_RE.findall(text)


def _is_word(token: str) -> bool:
    """A token counts as a 'word' if it contains a letter or digit."""
    return any(c.isalnum() for c in token)


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 1: spaCy noun_chunks (legacy, position-based)
# ─────────────────────────────────────────────────────────────────────────────

def extract_noun_slots(text: str) -> list[dict]:
    """Extract noun-phrase slots from a single text using spaCy noun_chunks.

    Empty chunks (e.g. bare 'that') are skipped.
    """
    if not SPACY_AVAILABLE or not text.strip():
        return []

    doc = _nlp(text)
    slots: list[dict] = []

    for chunk in doc.noun_chunks:
        nouns = [tok.text for tok in chunk if tok.pos_ in {"NOUN", "PROPN"}]
        if not nouns:
            continue
        slots.append({
            "phrase": chunk.text.strip(),
            "nouns": nouns,
            "noun_count": len(nouns),
        })

    return slots


# ─────────────────────────────────────────────────────────────────────────────
#  Strategy 2: diff-based slot extraction (auto-infers skeleton)
# ─────────────────────────────────────────────────────────────────────────────

def extract_diff_slots(text_a: str, text_b: str) -> dict:
    """Auto-infer the locked skeleton by diffing two parallel texts.

    Returns:
      {
        "skeleton": [<list of common tokens>],   # the locked words
        "slots": [
            {
              "index":    1,
              "phrase_a": "metabolic output",
              "phrase_b": "leg drive",
              "words_a":  2,
              "words_b":  2,
              "nouns_a":  1,
              "nouns_b":  2,
            }, ...
        ]
      }
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)

    skeleton: list[str] = []
    slots: list[dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            skeleton.extend(tokens_a[i1:i2])
            continue

        slot_tokens_a = tokens_a[i1:i2]
        slot_tokens_b = tokens_b[j1:j2]

        words_a = [t for t in slot_tokens_a if _is_word(t)]
        words_b = [t for t in slot_tokens_b if _is_word(t)]

        # Skip slots that are pure punctuation drift
        if not words_a and not words_b:
            continue

        phrase_a = " ".join(slot_tokens_a).strip()
        phrase_b = " ".join(slot_tokens_b).strip()

        slots.append({
            "index":    len(slots) + 1,
            "phrase_a": phrase_a,
            "phrase_b": phrase_b,
            "words_a":  len(words_a),
            "words_b":  len(words_b),
            "nouns_a":  _count_nouns(phrase_a),
            "nouns_b":  _count_nouns(phrase_b),
        })

    return {"skeleton": skeleton, "slots": slots}


def _count_nouns(text: str) -> int:
    if not SPACY_AVAILABLE or not text.strip():
        return 0
    return sum(1 for tok in _nlp(text) if tok.pos_ in {"NOUN", "PROPN"})


# ─────────────────────────────────────────────────────────────────────────────
#  Parity check (uses diff-based extraction)
# ─────────────────────────────────────────────────────────────────────────────

def check_noun_parity(nc_text: str, vc_text: str) -> tuple[bool, list[dict]]:
    """Rule 8 parity check using diff-inferred slots.

    Compares word counts per slot. Returns (passed, mismatches).
    """
    if not SPACY_AVAILABLE:
        return True, []

    diff = extract_diff_slots(nc_text, vc_text)
    mismatches: list[dict] = []

    for slot in diff["slots"]:
        if slot["words_a"] != slot["words_b"]:
            mismatches.append({
                "type":      "slot_mismatch",
                "slot":      slot["index"],
                "nc_phrase": slot["phrase_a"] or "(empty)",
                "nc_count":  slot["words_a"],
                "vc_phrase": slot["phrase_b"] or "(empty)",
                "vc_count":  slot["words_b"],
            })

    return len(mismatches) == 0, mismatches


def format_parity_feedback(mismatches: list[dict]) -> str:
    """Build an actionable retry message listing every mismatch."""
    if not mismatches:
        return ""

    lines = [
        "Your previous output FAILED Rule 8 (Word-Count Parity per Slot).",
        "These slots, where NC and VC differ from the locked skeleton, do NOT",
        "have matching word counts:",
        "",
    ]

    for m in mismatches:
        lines.append(
            f"- Slot {m['slot']}: NC '{m['nc_phrase']}' has {m['nc_count']} word"
            f"{'s' if m['nc_count'] != 1 else ''}, "
            f"but VC '{m['vc_phrase']}' has {m['vc_count']} word"
            f"{'s' if m['vc_count'] != 1 else ''}. "
            f"Both must contain the same number of words."
        )

    lines += [
        "",
        "REGENERATE all three stimuli (NC, MC, VC) keeping the locked sentence "
        "skeleton intact. Adjust ONLY the noun phrases so that every variable "
        "slot contains the same number of words across NC, MC, and VC. Do not "
        "change verbs, adjectives, articles, or sentence structure.",
    ]

    return "\n".join(lines)
