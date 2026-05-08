"""
Response parser: extracts NC / MC / VC blocks and sentences from raw Gemini output.
"""

import re


def extract_three_stimuli(raw: str) -> dict[str, str]:
    """Extract NC, MC, VC labeled blocks from model output."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return {"nc": "", "mc": "", "vc": ""}

    nc = _extract_labeled_block(text, "NC", ["MC", "VC"])
    mc = _extract_labeled_block(text, "MC", ["VC"])
    vc = _extract_labeled_block(text, "VC", [])
    return {"nc": nc, "mc": mc, "vc": vc}


def parse_sentences(text: str) -> list[str]:
    """Return clean sentence list from free-form text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        line = re.sub(r"^(\d+[\.\)]\s*|sentence\s+\d+[:\-\s]+)", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^[\-\*\#>\s]+", "", line).strip()
        if line:
            cleaned.append(line)

    joined = re.sub(r"\s+", " ", " ".join(cleaned)).strip()
    if not joined:
        return []

    parts = re.split(r"(?<=[.!?])\s+", joined)
    return [p.strip() for p in parts if p.strip()]


# ── Internal helpers ─────────────────────────────────────────────────────────

def _extract_labeled_block(text: str, label: str, next_labels: list[str]) -> str:
    if next_labels:
        next_alt = "|".join(rf"{re.escape(n)}\s*:" for n in next_labels)
        boundary = rf"^\s*(?:{next_alt})|\Z"
    else:
        boundary = r"\Z"
    pattern = rf"(?is)^\s*{re.escape(label)}\s*:\s*(.+?)(?={boundary})"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()
