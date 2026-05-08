"""
Image generation pipeline (paired NC+VC):
  1. Detect sentence position (S1–S5) via fingerprints
  2. Send BOTH the NC and VC versions of the sentence to Gemini together
     → get back a paired {nc_prompt, vc_prompt} JSON
  3. Send each prompt to gemini-3-pro-image-preview → get back an image
  4. Save everything for research tracking
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from google import genai

from config import IMAGE_BACKUP_DIR, GENERATED_IMAGES_DIR, BASE_DIR, get_gemini_key, IMAGEN_MODEL, IMAGEN_MODEL_FALLBACK, IMAGEN_MODEL_FALLBACK_2, task_slug, next_available_name
from logger import current_session

# ── Image generation timing log ───────────────────────────────────────────────
IMAGE_TIMING_LOG = BASE_DIR / "image_gen_timings.jsonl"
from image_system_prompt import IMAGE_SYSTEM_PROMPT
import gemini_client

# ── Fingerprints for sentence detection ──────────────────────────────────────

FINGERPRINTS = {
    "S1": ["can thoughtfully integrate"],
    "S2": ["designed to handle continuous", "seamlessly blend"],
    "S3": ["to ensure practical manufacturing", "can be configured to guarantee"],
    "S4": ["a highly adaptable", "allows users to translate"],
    "S5": ["ultimately, by exploring distinct", "truly affordable"],
}


def detect_position(sentence: str) -> str:
    lower = sentence.lower()
    for pos, phrases in FINGERPRINTS.items():
        if all(p in lower for p in phrases):
            return pos
    raise ValueError(f"Cannot identify sentence position: {sentence[:80]}...")


# ── Step 1: Gemini generates the paired (NC+VC) Imagen prompts ──────────────

def _parse_paired_json(raw: str) -> dict:
    """Extract {nc_prompt, vc_prompt} from Gemini's response, tolerant of
    code fences or surrounding text."""
    text = raw.strip()
    # Strip code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Try direct parse first
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: grab the first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise RuntimeError(f"Gemini did not return JSON: {raw[:300]}")
        obj = json.loads(match.group(0))

    nc = (obj.get("nc_prompt") or "").strip()
    vc = (obj.get("vc_prompt") or "").strip()
    if not nc or not vc:
        raise RuntimeError(f"Gemini JSON missing nc_prompt or vc_prompt: {raw[:300]}")
    return {"nc": nc, "vc": vc}


async def _get_paired_imagen_prompts(
    position: str, nc_sentence: str, vc_sentence: str
) -> dict:
    """Send both NC and VC sentences together, receive a paired JSON with
    two nearly-identical Imagen prompts that differ only in noun specificity."""
    user_msg = (
        f"Sentence position: {position}\n"
        f"NC sentence: {nc_sentence.strip()}\n"
        f"VC sentence: {vc_sentence.strip()}\n\n"
        f"Return ONE line of JSON: "
        f'{{"nc_prompt": "...", "vc_prompt": "..."}}. '
        f"Both prompts must share identical composition, lighting, framing, "
        f"and supporting elements, and differ ONLY in the specificity of the "
        f"primary objects."
    )

    raw = await gemini_client.generate(
        prompt=user_msg,
        system_prompt=IMAGE_SYSTEM_PROMPT,
        purpose="paired_prompts",
    )
    paired = _parse_paired_json(raw)

    sess = current_session()
    if sess is not None:
        sess.log_paired_prompts(
            position=position,
            nc_sentence=nc_sentence.strip(),
            vc_sentence=vc_sentence.strip(),
            nc_prompt=paired["nc"],
            vc_prompt=paired["vc"],
        )
    return paired


# ── Step 2: Google Imagen generates the image ───────────────────────────────

def _build_imagen_client() -> genai.Client:
    key = get_gemini_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    return genai.Client(api_key=key)


def _log_image_timing(duration: float, success: bool, model: str) -> None:
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(duration, 2),
        "success": success,
        "model": model,
    }
    with open(IMAGE_TIMING_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _generate_image_with_model(client: genai.Client, prompt: str, model: str) -> bytes:
    """Generate image using specified model."""
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )
    if not response.candidates or not response.candidates[0].content.parts:
        raise RuntimeError(f"{model} returned no image. The prompt may have been blocked.")
    part = response.candidates[0].content.parts[0]
    if not hasattr(part, "inline_data") or part.inline_data is None:
        raise RuntimeError(f"{model} response did not contain image data.")
    return part.inline_data.data


# Sticky preferred Imagen model. Starts at primary; whenever a call succeeds
# with a different model (because the preferred one failed), this is updated
# so future calls skip the broken model. Atomic writes — safe for parallel
# image calls since they all converge on the same successful model.
_preferred_imagen_model: str = IMAGEN_MODEL


def _build_imagen_try_order(start_model: str) -> list[str]:
    """Preferred model first, then the other configured Imagen models."""
    all_models = [IMAGEN_MODEL, IMAGEN_MODEL_FALLBACK, IMAGEN_MODEL_FALLBACK_2]
    return [start_model] + [m for m in all_models if m != start_model]


def _generate_image(prompt: str, position: str = "?", level: str = "?") -> bytes:
    """Generate image starting with the sticky preferred model. On success,
    pin the working model so subsequent calls skip the broken one.

    `position` ('S1'..'S5') and `level` ('nc' | 'vc') are recorded in the
    per-request log for traceability.
    """
    global _preferred_imagen_model
    client = _build_imagen_client()
    models_to_try = _build_imagen_try_order(_preferred_imagen_model)
    errors = []
    sess = current_session()

    for i, model in enumerate(models_to_try):
        start = time.monotonic()
        try:
            data = _generate_image_with_model(client, prompt, model)
            elapsed = time.monotonic() - start
            _log_image_timing(elapsed, success=True, model=model)
            if sess is not None:
                sess.log_image_call(
                    position=position, level=level, model=model,
                    duration_s=elapsed, success=True,
                    prompt_chars=len(prompt), size_bytes=len(data),
                )
            if model != _preferred_imagen_model:
                print(f"[image_gen] Preferred Imagen model: {_preferred_imagen_model} → {model} (sticky)")
                _preferred_imagen_model = model
            return data
        except Exception as e:
            elapsed = time.monotonic() - start
            _log_image_timing(elapsed, success=False, model=model)
            error_msg = str(e)
            if sess is not None:
                sess.log_image_call(
                    position=position, level=level, model=model,
                    duration_s=elapsed, success=False,
                    prompt_chars=len(prompt), error=error_msg[:500],
                )
            errors.append(f"{model}: {error_msg}")
            if i < len(models_to_try) - 1:
                print(f"[image_gen] {model} failed: {error_msg}, trying next model...")

    raise RuntimeError(
        "All image models failed.\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


# ── Public API ───────────────────────────────────────────────────────────────

async def generate_paired_images(
    nc_text: str,
    vc_text: str,
    nc_sentences: list[str],
    vc_sentences: list[str],
    task: str,
    requirements: list[str],
) -> dict:
    """
    Generate paired NC + VC images for all 5 sentence positions in parallel.

    For each sentence position (S1–S5):
      - One Gemini call receives BOTH the NC and VC sentences together and
        returns a paired JSON of two Imagen prompts sharing composition,
        differing only in object specificity.
      - Two image calls (NC and VC) fire in parallel.

    Returns:
      {
        "nc": [ {position, sentence, imagen_prompt, filename, image_url}, ... ],
        "vc": [ {position, sentence, imagen_prompt, filename, image_url}, ... ],
      }
    """
    if len(nc_sentences) != len(vc_sentences):
        raise ValueError(
            f"NC and VC must have the same number of sentences "
            f"(got {len(nc_sentences)} and {len(vc_sentences)})"
        )

    slug = task_slug(task)
    backup_folder = next_available_name(IMAGE_BACKUP_DIR, slug)
    backup_dir = IMAGE_BACKUP_DIR / backup_folder
    (backup_dir / "nc").mkdir(parents=True, exist_ok=True)
    (backup_dir / "vc").mkdir(parents=True, exist_ok=True)

    async def _process_position(i: int, nc_sent: str, vc_sent: str) -> dict | None:
        nc_sent = nc_sent.strip()
        vc_sent = vc_sent.strip()
        if not nc_sent or not vc_sent:
            return None

        # Detect position from whichever sentence fingerprints cleanly
        try:
            position = detect_position(vc_sent)
        except ValueError:
            try:
                position = detect_position(nc_sent)
            except ValueError:
                position = f"S{i + 1}"

        # Step 1: ONE Gemini call → paired (nc_prompt, vc_prompt)
        paired = await _get_paired_imagen_prompts(position, nc_sent, vc_sent)

        # Step 2: TWO parallel image calls
        nc_bytes, vc_bytes = await asyncio.gather(
            asyncio.to_thread(_generate_image, paired["nc"], position, "nc"),
            asyncio.to_thread(_generate_image, paired["vc"], position, "vc"),
        )

        # Save to served directory + backup, for each level
        def _save(level: str, prompt: str, img_bytes: bytes, sentence: str) -> dict:
            img_base = f"{backup_folder}_{level}_{position}"
            filename = next_available_name(GENERATED_IMAGES_DIR, img_base, ".png")
            (GENERATED_IMAGES_DIR / filename).write_bytes(img_bytes)
            (backup_dir / level / f"{position}.png").write_bytes(img_bytes)
            return {
                "position": position,
                "sentence": sentence,
                "imagen_prompt": prompt,
                "filename": filename,
                "image_url": f"/images/{filename}",
            }

        return {
            "nc": _save("nc", paired["nc"], nc_bytes, nc_sent),
            "vc": _save("vc", paired["vc"], vc_bytes, vc_sent),
            "position": position,
        }

    # Run all 5 positions in parallel
    raw_results = await asyncio.gather(
        *[_process_position(i, nc, vc) for i, (nc, vc) in enumerate(zip(nc_sentences, vc_sentences))]
    )
    results = [r for r in raw_results if r is not None]

    nc_results = [r["nc"] for r in results]
    vc_results = [r["vc"] for r in results]

    # Save prompts.txt for each level
    def _write_prompts(level: str, items: list[dict]) -> None:
        lines = []
        for p in items:
            lines.append(f"=== {p['position']} ===")
            lines.append(f"Sentence: {p['sentence']}")
            lines.append(f"Imagen Prompt: {p['imagen_prompt']}")
            lines.append("")
        (backup_dir / level / "prompts.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    _write_prompts("nc", nc_results)
    _write_prompts("vc", vc_results)

    (backup_dir / "nc" / "stimulus.txt").write_text(nc_text, encoding="utf-8")
    (backup_dir / "vc" / "stimulus.txt").write_text(vc_text, encoding="utf-8")

    # Save unified metadata
    meta = {
        "session_id": backup_folder,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_prompt_gen": "gemini (paired via image_system_prompt)",
        "model_image_gen": IMAGEN_MODEL,
        "task": task,
        "requirements": requirements,
        "nc": {
            "stimulus_text": nc_text,
            "sentences": nc_sentences,
            "images": [
                {"position": r["position"], "sentence": r["sentence"],
                 "imagen_prompt": r["imagen_prompt"], "filename": r["filename"]}
                for r in nc_results
            ],
        },
        "vc": {
            "stimulus_text": vc_text,
            "sentences": vc_sentences,
            "images": [
                {"position": r["position"], "sentence": r["sentence"],
                 "imagen_prompt": r["imagen_prompt"], "filename": r["filename"]}
                for r in vc_results
            ],
        },
    }
    (backup_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {"nc": nc_results, "vc": vc_results}
