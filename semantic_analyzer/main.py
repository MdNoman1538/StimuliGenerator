"""
Semantic Abstraction Comparator
ââââââââââââââââââââââââââââââââ
Extracts all nouns from Text A and Text B, then scores each noun with
WordNet abstraction level.
Results saved to .mat files.
"""

import os, re, difflib, datetime
import numpy as np
import scipy.io
import scipy.stats
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from brysbaert_data import get_concreteness, normalize_concreteness

# ââ spaCy ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception as e:
    SPACY_OK = False; SPACY_ERR = str(e)

# ââ NLTK / WordNet ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
try:
    import nltk
    from nltk.corpus import wordnet as wn
    auto_download = os.getenv("NLTK_AUTO_DOWNLOAD", "").strip().lower() in {"1", "true", "yes"}
    if auto_download:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    try:
        wn.ensure_loaded()
        NLTK_OK = True
    except LookupError:
        NLTK_OK = False
        NLTK_ERR = "WordNet data not available. Run: python -m nltk.downloader wordnet omw-1.4"
except Exception as e:
    NLTK_OK = False; NLTK_ERR = str(e)

OUTPUT_DIR = "output_mat"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_WN_DEPTH = 20

app = FastAPI(title="Semantic Abstraction Comparator")


# ââ Exception handler for runtime errors

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  ABSTRACTION LEVEL  (WordNet hypernym depth, normalised to [0,1])
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def wn_depth_min(word: str) -> int | None:
    """Shortest path from any sense of `word` to WordNet root (most abstract reading)."""
    try:
        syns = wn.synsets(word, pos=wn.NOUN)
    except LookupError:
        return None
    if not syns:
        return None
    depths = []
    for s in syns:
        paths = s.hypernym_paths()
        if paths:
            depths.append(min(len(p) for p in paths))
    return min(depths) if depths else None


def wn_depth_mean(word: str) -> float | None:
    """Mean path length across ALL senses and ALL hypernym paths (ontological position)."""
    try:
        syns = wn.synsets(word, pos=wn.NOUN)
    except LookupError:
        return None
    if not syns:
        return None
    lengths = [len(p) for s in syns for p in s.hypernym_paths()]
    if not lengths:
        return None
    return float(np.mean(lengths))


def wn_depth(word: str, formula: str = "wordnet_min") -> int | float | None:
    """Public depth dispatcher for WordNet.
    
    formula = 'wordnet_min' (default), 'wordnet_mean', or legacy names 'min', 'mean'
    """
    formula = formula.lower()
    if formula in ("mean", "wordnet_mean"):
        return wn_depth_mean(word)
    return wn_depth_min(word)


def abstraction_score(word: str, formula: str = "wordnet_min") -> float | None:
    """Abstraction Level (AL) calculation.
    
    Available formulas:
    - 'wordnet_min'     → shortest hypernym path (most-abstract sense, default)
    - 'wordnet_mean'    → mean path length across all senses (ontological hierarchy)
    - 'brysbaert'       → Brysbaert Concreteness Index (1–5 scale, normalized to AL)
    
    All return scores in [0, 1] where 1 = most abstract, 0 = most concrete.
    """
    formula = formula.lower()
    
    # Brysbaert Concreteness Index
    if formula == "brysbaert":
        concrete_score = get_concreteness(word)
        if concrete_score is None:
            return None
        al = normalize_concreteness(concrete_score, min_val=1.0, max_val=5.0)
        return round(al, 4) if al is not None else None
    
    # WordNet-based formulas (backward compatible with old names)
    if formula in ("min", "wordnet_min", "wordnet_mean", "mean"):
        d = wn_depth(word, formula=formula)
        if d is None:
            return None
        return round(float(np.clip(1.0 - (d - 1) / (MAX_WN_DEPTH - 1), 0.0, 1.0)), 4)
    
    # Default to WordNet minimum if formula not recognized
    d = wn_depth(word, formula="wordnet_min")
    if d is None:
        return None
    return round(float(np.clip(1.0 - (d - 1) / (MAX_WN_DEPTH - 1), 0.0, 1.0)), 4)


def extract_all_nouns(text: str) -> list[tuple[str, str]]:
    """Return all noun occurrences as (surface_form, lemma)."""
    if not SPACY_OK:
        raise RuntimeError("spaCy model is not available; cannot extract nouns.")
    doc = nlp(text)
    out: list[tuple[str, str]] = []
    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN"}:
            surface = tok.text.strip()
            lemma = tok.lemma_.lower().strip() if tok.lemma_ else tok.text.lower().strip()
            if surface:
                out.append((surface, lemma))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Diff-based slot extraction (auto-infers locked skeleton)
# ─────────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b[\w'-]+\b|[^\w\s]")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _is_word(tok: str) -> bool:
    return any(c.isalnum() for c in tok)


def count_words(text: str) -> int:
    """Total word count (excludes pure-punctuation tokens)."""
    return sum(1 for tok in _tokenize(text) if _is_word(tok))


# ─────────────────────────────────────────────────────────────────────────────
#  Paired t-test on the 25 slot AL values (NC vs VC)
# ─────────────────────────────────────────────────────────────────────────────

def paired_ttest(values_a: list[float | None],
                 values_b: list[float | None]) -> dict:
    """Run a paired t-test on aligned per-slot abstraction values.

    Pairs where either side is None (OOV / missing AL) are excluded.
    Returns descriptive stats + t-test outputs, or `n < 2` gives a null result.
    """
    pairs = [(a, b) for a, b in zip(values_a, values_b)
             if a is not None and b is not None]
    n = len(pairs)
    excluded = len(values_a) - n

    base = {
        "n":             n,
        "n_excluded":    excluded,
        "df":            None,
        "t_statistic":   None,
        "p_value":       None,
        "mean_diff":     None,
        "std_diff":      None,
        "ci95_low":      None,
        "ci95_high":     None,
        "interpretation": "insufficient data" if n < 2 else "",
    }
    if n < 2:
        return base

    a_arr = np.array([p[0] for p in pairs], dtype=float)
    b_arr = np.array([p[1] for p in pairs], dtype=float)
    diffs = a_arr - b_arr

    res = scipy.stats.ttest_rel(a_arr, b_arr)
    t_stat = float(res.statistic)
    p_val  = float(res.pvalue)
    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    sem       = std_diff / np.sqrt(n) if n > 1 else 0.0
    # 95% CI via t-distribution
    t_crit = scipy.stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
    ci_low  = mean_diff - t_crit * sem
    ci_high = mean_diff + t_crit * sem

    if p_val < 0.001:
        sig_label = "p < 0.001"
    elif p_val < 0.01:
        sig_label = "p < 0.01"
    elif p_val < 0.05:
        sig_label = "p < 0.05"
    else:
        sig_label = "n.s."

    if p_val < 0.05:
        direction = ("NC is significantly more abstract than VC"
                     if mean_diff > 0
                     else "VC is significantly more abstract than NC")
    else:
        direction = "no significant difference between NC and VC"

    base.update({
        "df":             n - 1,
        "t_statistic":    round(t_stat, 4),
        "p_value":        round(p_val, 6),
        "mean_diff":      round(mean_diff, 4),
        "std_diff":       round(std_diff, 4),
        "ci95_low":       round(float(ci_low), 4),
        "ci95_high":      round(float(ci_high), 4),
        "sig_label":      sig_label,
        "interpretation": f"{direction} ({sig_label})",
    })
    return base


def independent_ttest(values_a: list[float | None],
                      values_b: list[float | None]) -> dict:
    """Welch's t-test on two independent samples of AL values (unequal n / variance)."""
    a = [v for v in values_a if v is not None]
    b = [v for v in values_b if v is not None]
    n_a, n_b = len(a), len(b)
    base = {
        "n_a":            n_a,
        "n_b":            n_b,
        "df":             None,
        "t_statistic":    None,
        "p_value":        None,
        "mean_a":         None,
        "mean_b":         None,
        "mean_diff":      None,
        "sig_label":      None,
        "interpretation": "insufficient data" if (n_a < 2 or n_b < 2) else "",
    }
    if n_a < 2 or n_b < 2:
        return base

    res = scipy.stats.ttest_ind(a, b, equal_var=False)
    t_stat = float(res.statistic)
    p_val  = float(res.pvalue)
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    mean_diff = mean_a - mean_b
    df = float(getattr(res, "df", n_a + n_b - 2))

    if p_val < 0.001:
        sig_label = "p < 0.001"
    elif p_val < 0.01:
        sig_label = "p < 0.01"
    elif p_val < 0.05:
        sig_label = "p < 0.05"
    else:
        sig_label = "n.s."

    if p_val < 0.05:
        direction = ("text A is significantly more abstract than text B"
                     if mean_diff > 0
                     else "text B is significantly more abstract than text A")
    else:
        direction = "no significant difference between text A and text B"

    base.update({
        "df":             round(df, 2),
        "t_statistic":    round(t_stat, 4),
        "p_value":        round(p_val, 6),
        "mean_a":         round(mean_a, 4),
        "mean_b":         round(mean_b, 4),
        "mean_diff":      round(mean_diff, 4),
        "sig_label":      sig_label,
        "interpretation": f"{direction} ({sig_label})",
    })
    return base


def _analyze_text_for_general(text: str, formula: str = "min") -> dict:
    """Extract every noun individually + score abstraction. No alignment."""
    text = text.strip()
    if not SPACY_OK or not text:
        return {"nouns": [], "word_count": 0, "noun_count": 0,
                "wn_count": 0, "avg_al": None}

    word_count = count_words(text)
    nouns_raw = extract_all_nouns(text)
    entries: list[dict] = []
    scores: list[float] = []
    wn_count = 0

    for surface, lemma in nouns_raw:
        depth    = wn_depth(lemma, formula=formula)
        al       = abstraction_score(lemma, formula=formula)
        conc_raw = get_concreteness(lemma) if formula == "brysbaert" else None
        in_wn    = (al is not None)
        if in_wn:
            wn_count += 1
        if al is not None:
            scores.append(al)
        entries.append({
            "noun":      surface,
            "lemma":     lemma,
            "depth":     depth if (isinstance(depth, int) or depth is None) else round(depth, 2),
            "al":        al,
            "conc_raw":  round(conc_raw, 2) if conc_raw is not None else None,
            "in_wordnet": in_wn,
        })

    avg_al = round(float(np.mean(scores)), 4) if scores else None
    return {
        "nouns":       entries,
        "word_count":  word_count,
        "noun_count":  len(entries),
        "wn_count":    wn_count,
        "avg_al":      avg_al,
    }


def _analyze_phrase(phrase: str, formula: str = "min") -> dict:
    """Extract nouns + WordNet abstraction scores from a phrase string."""
    if not SPACY_OK or not phrase.strip():
        return {"nouns": [], "lemmas": [], "depths": [], "scores": [],
                "raw_scores": [], "avg_al": None, "wn_count": 0}
    doc = nlp(phrase)
    nouns, lemmas, depths, scores, raw_scores = [], [], [], [], []
    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN"}:
            lemma = tok.lemma_.lower().strip() if tok.lemma_ else tok.text.lower().strip()
            nouns.append(tok.text.strip())
            lemmas.append(lemma)
            depths.append(wn_depth(lemma, formula=formula))
            scores.append(abstraction_score(lemma, formula=formula))
            raw_scores.append(get_concreteness(lemma) if formula == "brysbaert" else None)
    valid = [s for s in scores if s is not None]
    avg = round(float(np.mean(valid)), 4) if valid else None
    return {
        "nouns": nouns, "lemmas": lemmas, "depths": depths, "scores": scores,
        "raw_scores": raw_scores,
        "avg_al": avg, "wn_count": sum(1 for s in scores if s is not None),
    }


def extract_diff_slots(text_a: str, text_b: str, formula: str = "min") -> dict:
    """Auto-infer the locked skeleton by diffing two parallel texts.

    Returns paired slots (phrase_a vs phrase_b) where the texts differ from
    the shared skeleton — same number of slots in both, perfectly aligned.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b, autojunk=False)

    skeleton: list[str] = []
    pairs: list[dict] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            skeleton.extend(tokens_a[i1:i2])
            continue
        slot_a_toks = tokens_a[i1:i2]
        slot_b_toks = tokens_b[j1:j2]
        words_a = [t for t in slot_a_toks if _is_word(t)]
        words_b = [t for t in slot_b_toks if _is_word(t)]
        if not words_a and not words_b:
            continue
        phrase_a = " ".join(slot_a_toks).strip()
        phrase_b = " ".join(slot_b_toks).strip()
        info_a = _analyze_phrase(phrase_a, formula=formula)
        info_b = _analyze_phrase(phrase_b, formula=formula)
        pairs.append({
            "index":    len(pairs) + 1,
            "phrase_a": phrase_a,
            "phrase_b": phrase_b,
            "words_a":  len(words_a),
            "words_b":  len(words_b),
            **{f"{k}_a": v for k, v in info_a.items()},
            **{f"{k}_b": v for k, v in info_b.items()},
        })

    return {"skeleton": skeleton, "pairs": pairs}


def extract_noun_slots(text: str, formula: str = "wordnet_min") -> list[dict]:
    """
    Extract ordered noun-phrase slots.
    Each slot contains all nouns found inside the chunk, allowing
    per-slot averaging when one side has multiple nouns.
    """
    if not SPACY_OK:
        raise RuntimeError("spaCy model is not available; cannot extract noun slots.")

    doc = nlp(text)
    slots: list[dict] = []

    for chunk in doc.noun_chunks:
        nouns: list[str] = []
        lemmas: list[str] = []
        depths: list[int | None] = []
        scores: list[float | None] = []

        for tok in chunk:
            if tok.pos_ in {"NOUN", "PROPN"}:
                noun = tok.text.strip()
                lemma = tok.lemma_.lower().strip() if tok.lemma_ else tok.text.lower().strip()
                nouns.append(noun)
                lemmas.append(lemma)
                # For Brysbaert, depth is not applicable; for WordNet, we still compute it
                if formula.lower() == "brysbaert":
                    depth = None
                else:
                    depth = wn_depth(lemma, formula=formula)
                depths.append(depth)
                scores.append(abstraction_score(lemma, formula=formula))

        if not nouns:
            continue

        valid_scores = [s for s in scores if s is not None]
        avg_score = round(float(np.mean(valid_scores)), 4) if valid_scores else None

        slots.append({
            "phrase": chunk.text.strip(),
            "nouns": nouns,
            "lemmas": lemmas,
            "depths": depths,
            "scores": scores,
            "avg_al": avg_score,
            "wn_count": sum(1 for d in depths if d is not None),
        })

    # Fallback to single nouns if noun chunks are empty.
    if not slots:
        for noun, lemma in extract_all_nouns(text):
            if formula.lower() == "brysbaert":
                depth = None
            else:
                depth = wn_depth(lemma, formula=formula)
            score = abstraction_score(lemma, formula=formula)
            slots.append({
                "phrase": noun,
                "nouns": [noun],
                "lemmas": [lemma],
                "depths": [depth],
                "scores": [score],
                "avg_al": score,
                "wn_count": 1 if depth is not None else 0,
            })

    return slots


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  PYDANTIC MODELS
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class CompareRequest(BaseModel):
    text_a: str
    text_b: str
    label: str = "comparison"
    # AL formula options:
    #   'wordnet_min'  → shortest hypernym path (most-abstract sense, default)
    #   'wordnet_mean' → mean path length across all senses (ontological hierarchy)
    #   'brysbaert'    → Brysbaert Concreteness Index (psycholinguistic ratings)
    # Legacy names still supported: 'min' → 'wordnet_min', 'mean' → 'wordnet_mean'
    formula: str = "wordnet_min"


class SlotScore(BaseModel):
    slot:           int
    phrase_a:       str | None
    phrase_b:       str | None
    nouns_a:        list[str]
    nouns_b:        list[str]
    avg_al_a:       float | None
    avg_al_b:       float | None
    delta:          float | None
    wn_a:           int
    wn_b:           int


class TTestResult(BaseModel):
    n:              int
    n_excluded:     int
    df:             int | None
    t_statistic:    float | None
    p_value:        float | None
    mean_diff:      float | None
    std_diff:       float | None
    ci95_low:       float | None
    ci95_high:      float | None
    sig_label:      str | None = None
    interpretation: str


class IndepTTestResult(BaseModel):
    n_a:             int
    n_b:             int
    df:              float | None
    t_statistic:     float | None
    p_value:         float | None
    mean_a:          float | None
    mean_b:          float | None
    mean_diff:       float | None
    interpretation:  str
    sig_label:       str | None = None


class GeneralNounEntry(BaseModel):
    noun:        str
    lemma:       str
    depth:       float | None  # int for 'min', float for 'mean'
    al:          float | None
    in_wordnet:  bool


class GeneralSide(BaseModel):
    nouns:       list[GeneralNounEntry]
    word_count:  int
    noun_count:  int
    wn_count:    int
    avg_al:      float | None


class GeneralCompareResponse(BaseModel):
    label:    str
    formula:  str
    text_a:   GeneralSide
    text_b:   GeneralSide
    ttest:    IndepTTestResult
    mat_file: str


class CompareResponse(BaseModel):
    label:        str
    formula:      str
    slots:        list[SlotScore]
    word_count_a: int
    word_count_b: int
    noun_count_a: int
    noun_count_b: int
    avg_al_a:     float | None
    avg_al_b:     float | None
    ttest:        TTestResult
    mat_file:     str


# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  ENDPOINTS
# ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

@app.post("/compare", response_model=CompareResponse)
async def compare(req: CompareRequest):
    """Diff-based comparison: auto-infers the locked skeleton, pairs slots
    where the two texts differ, and scores each slot's nouns via WordNet or Brysbaert."""
    label = re.sub(r"[^\w\-]", "_", req.label.strip()) or "comparison"

    # Validate and normalize formula
    formula_lower = req.formula.lower()
    valid_formulas = {"wordnet_min", "wordnet_mean", "brysbaert", "min", "mean"}
    if formula_lower not in valid_formulas:
        formula = "wordnet_min"
    else:
        # Normalize old-style names to new names
        formula = "wordnet_min" if formula_lower == "min" else (
                  "wordnet_mean" if formula_lower == "mean" else formula_lower)
    
    text_a = req.text_a.strip()
    text_b = req.text_b.strip()
    word_count_a = count_words(text_a)
    word_count_b = count_words(text_b)

    diff = extract_diff_slots(text_a, text_b, formula=formula)
    pairs = diff["pairs"]

    # Aggregate scores across all slots (only words that have WordNet entries)
    all_scores_a = [s for p in pairs for s in p["scores_a"] if s is not None]
    all_scores_b = [s for p in pairs for s in p["scores_b"] if s is not None]
    avg_a = round(float(np.mean(all_scores_a)), 4) if all_scores_a else None
    avg_b = round(float(np.mean(all_scores_b)), 4) if all_scores_b else None

    all_raw_a = [r for p in pairs for r in p.get("raw_scores_a", []) if r is not None]
    all_raw_b = [r for p in pairs for r in p.get("raw_scores_b", []) if r is not None]
    avg_raw_a = round(float(np.mean(all_raw_a)), 4) if all_raw_a else None
    avg_raw_b = round(float(np.mean(all_raw_b)), 4) if all_raw_b else None

    slots: list[SlotScore] = []
    for p in pairs:
        delta = (round(p["avg_al_a"] - p["avg_al_b"], 4)
                 if p["avg_al_a"] is not None and p["avg_al_b"] is not None else None)
        slots.append(SlotScore(
            slot=p["index"],
            phrase_a=p["phrase_a"] or None,
            phrase_b=p["phrase_b"] or None,
            nouns_a=p["nouns_a"],
            nouns_b=p["nouns_b"],
            avg_al_a=p["avg_al_a"],
            avg_al_b=p["avg_al_b"],
            delta=delta,
            wn_a=p["wn_count_a"],
            wn_b=p["wn_count_b"],
        ))

    all_nouns_a  = [n for p in pairs for n in p["nouns_a"]]
    all_nouns_b  = [n for p in pairs for n in p["nouns_b"]]
    all_lemmas_a = [l for p in pairs for l in p["lemmas_a"]]
    all_lemmas_b = [l for p in pairs for l in p["lemmas_b"]]

    # Paired t-test on the per-slot AL values
    slot_al_a = [p["avg_al_a"] for p in pairs]
    slot_al_b = [p["avg_al_b"] for p in pairs]
    ttest = paired_ttest(slot_al_a, slot_al_b)

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mat_name = f"{label}_{ts}.mat"
    mat_path = os.path.join(OUTPUT_DIR, mat_name)

    def _arr(lst, default=np.nan):
        return np.array([v if v is not None else default for v in lst])

    scipy.io.savemat(mat_path, {
        "label":         label,
        "formula":       formula,
        "text_a":        req.text_a,
        "text_b":        req.text_b,
        "word_count_a":  np.array([word_count_a], dtype=np.int32),
        "word_count_b":  np.array([word_count_b], dtype=np.int32),
        "skeleton":      np.array(diff["skeleton"], dtype=object),
        "all_nouns_a":   np.array(all_nouns_a, dtype=object),
        "all_nouns_b":   np.array(all_nouns_b, dtype=object),
        "all_lemmas_a":  np.array(all_lemmas_a, dtype=object),
        "all_lemmas_b":  np.array(all_lemmas_b, dtype=object),
        "phrases_a":     np.array([p["phrase_a"] for p in pairs], dtype=object),
        "phrases_b":     np.array([p["phrase_b"] for p in pairs], dtype=object),
        "slot_words_a":  np.array([p["words_a"] for p in pairs], dtype=np.int32),
        "slot_words_b":  np.array([p["words_b"] for p in pairs], dtype=np.int32),
        "slot_nouns_a":  np.array([" | ".join(p["nouns_a"]) for p in pairs], dtype=object),
        "slot_nouns_b":  np.array([" | ".join(p["nouns_b"]) for p in pairs], dtype=object),
        "avg_slot_al_a": _arr([p["avg_al_a"] for p in pairs]),
        "avg_slot_al_b": _arr([p["avg_al_b"] for p in pairs]),
        "all_al_a":      _arr(all_scores_a),
        "all_al_b":      _arr(all_scores_b),
        "avg_al_a":      np.array([avg_a if avg_a is not None else np.nan]),
        "avg_al_b":      np.array([avg_b if avg_b is not None else np.nan]),
        # Brysbaert raw concreteness scores (1=abstract, 5=concrete); NaN when not applicable
        "all_conc_raw_a":  _arr([r for p in pairs for r in p.get("raw_scores_a", [])]),
        "all_conc_raw_b":  _arr([r for p in pairs for r in p.get("raw_scores_b", [])]),
        "avg_conc_raw_a":  np.array([avg_raw_a if avg_raw_a is not None else np.nan]),
        "avg_conc_raw_b":  np.array([avg_raw_b if avg_raw_b is not None else np.nan]),
        "wn_slot_a":     np.array([p["wn_count_a"] for p in pairs], dtype=np.int32),
        "wn_slot_b":     np.array([p["wn_count_b"] for p in pairs], dtype=np.int32),
        "ttest_n":          np.array([ttest["n"]], dtype=np.int32),
        "ttest_df":         np.array([ttest["df"] if ttest["df"] is not None else -1], dtype=np.int32),
        "ttest_t":          np.array([ttest["t_statistic"] if ttest["t_statistic"] is not None else np.nan]),
        "ttest_p":          np.array([ttest["p_value"] if ttest["p_value"] is not None else np.nan]),
        "ttest_mean_diff":  np.array([ttest["mean_diff"] if ttest["mean_diff"] is not None else np.nan]),
        "ttest_std_diff":   np.array([ttest["std_diff"] if ttest["std_diff"] is not None else np.nan]),
        "ttest_ci95_low":   np.array([ttest["ci95_low"] if ttest["ci95_low"] is not None else np.nan]),
        "ttest_ci95_high":  np.array([ttest["ci95_high"] if ttest["ci95_high"] is not None else np.nan]),
    })

    return CompareResponse(
        label=label, formula=formula, slots=slots,
        word_count_a=word_count_a,
        word_count_b=word_count_b,
        noun_count_a=len(all_nouns_a),
        noun_count_b=len(all_nouns_b),
        avg_al_a=avg_a, avg_al_b=avg_b,
        ttest=TTestResult(**ttest),
        mat_file=mat_path,
    )


@app.post("/compare-general", response_model=GeneralCompareResponse)
async def compare_general(req: CompareRequest):
    """General comparison: extract every noun from each text independently,
    score with WordNet or Brysbaert, run an independent-samples t-test on the AL values.
    No skeleton, no slot pairing — for unrelated texts."""
    label = re.sub(r"[^\w\-]", "_", req.label.strip()) or "comparison"
    
    # Validate and normalize formula
    formula_lower = req.formula.lower()
    valid_formulas = {"wordnet_min", "wordnet_mean", "brysbaert", "min", "mean"}
    if formula_lower not in valid_formulas:
        formula = "wordnet_min"
    else:
        # Normalize old-style names to new names
        formula = "wordnet_min" if formula_lower == "min" else (
                  "wordnet_mean" if formula_lower == "mean" else formula_lower)

    side_a = _analyze_text_for_general(req.text_a, formula=formula)
    side_b = _analyze_text_for_general(req.text_b, formula=formula)

    ttest = independent_ttest(
        [n["al"] for n in side_a["nouns"]],
        [n["al"] for n in side_b["nouns"]],
    )

    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mat_name = f"{label}_general_{ts}.mat"
    mat_path = os.path.join(OUTPUT_DIR, mat_name)

    def _arr(lst, default=np.nan):
        return np.array([v if v is not None else default for v in lst])

    scipy.io.savemat(mat_path, {
        "label":       label,
        "formula":     formula,
        "mode":        "general",
        "text_a":      req.text_a,
        "text_b":      req.text_b,
        "nouns_a":     np.array([n["noun"]  for n in side_a["nouns"]], dtype=object),
        "nouns_b":     np.array([n["noun"]  for n in side_b["nouns"]], dtype=object),
        "lemmas_a":    np.array([n["lemma"] for n in side_a["nouns"]], dtype=object),
        "lemmas_b":    np.array([n["lemma"] for n in side_b["nouns"]], dtype=object),
        "al_a":           _arr([n["al"]       for n in side_a["nouns"]]),
        "al_b":           _arr([n["al"]       for n in side_b["nouns"]]),
        "conc_raw_a":     _arr([n["conc_raw"] for n in side_a["nouns"]]),
        "conc_raw_b":     _arr([n["conc_raw"] for n in side_b["nouns"]]),
        "depth_a":        _arr([n["depth"]    for n in side_a["nouns"]], default=-1),
        "depth_b":        _arr([n["depth"]    for n in side_b["nouns"]], default=-1),
        "word_count_a": np.array([side_a["word_count"]], dtype=np.int32),
        "word_count_b": np.array([side_b["word_count"]], dtype=np.int32),
        "noun_count_a": np.array([side_a["noun_count"]], dtype=np.int32),
        "noun_count_b": np.array([side_b["noun_count"]], dtype=np.int32),
        "wn_count_a":   np.array([side_a["wn_count"]], dtype=np.int32),
        "wn_count_b":   np.array([side_b["wn_count"]], dtype=np.int32),
        "avg_al_a":       np.array([side_a["avg_al"] if side_a["avg_al"] is not None else np.nan]),
        "avg_al_b":       np.array([side_b["avg_al"] if side_b["avg_al"] is not None else np.nan]),
        # Brysbaert raw concreteness scores (1=abstract, 5=concrete); NaN when not applicable
        "avg_conc_raw_a": np.array([np.nanmean(_arr([n["conc_raw"] for n in side_a["nouns"]])) if any(n["conc_raw"] is not None for n in side_a["nouns"]) else np.nan]),
        "avg_conc_raw_b": np.array([np.nanmean(_arr([n["conc_raw"] for n in side_b["nouns"]])) if any(n["conc_raw"] is not None for n in side_b["nouns"]) else np.nan]),
        "ttest_n_a":   np.array([ttest["n_a"]], dtype=np.int32),
        "ttest_n_b":   np.array([ttest["n_b"]], dtype=np.int32),
        "ttest_t":     np.array([ttest["t_statistic"] if ttest["t_statistic"] is not None else np.nan]),
        "ttest_p":     np.array([ttest["p_value"] if ttest["p_value"] is not None else np.nan]),
    })

    return GeneralCompareResponse(
        label=label, formula=formula,
        text_a=GeneralSide(**side_a),
        text_b=GeneralSide(**side_b),
        ttest=IndepTTestResult(**ttest),
        mat_file=mat_path,
    )


@app.get("/status")
async def status():
    return {
        "spacy":  SPACY_OK  if SPACY_OK  else f"ERROR: {SPACY_ERR}",
        "nltk":   NLTK_OK   if NLTK_OK   else f"ERROR: {NLTK_ERR}",
        "analyzer": SPACY_OK and NLTK_OK,
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    reload_enabled = os.getenv("UVICORN_RELOAD", "").strip().lower() in {"1", "true", "yes"}
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("main:app", host=host, port=port, reload=reload_enabled)
