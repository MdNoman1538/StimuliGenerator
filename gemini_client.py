"""
Gemini API client. Single responsibility: send a prompt, get text back.
Tries the primary model first, then a fallback if the primary fails.
"""

import asyncio
import time
from google import genai

from config import (
    get_gemini_key, GEMINI_MODEL, GEMINI_MODEL_FALLBACK, GEMINI_MODEL_FALLBACK_2,
    MAX_OUTPUT_TOKENS, API_TIMEOUT_S, FALLBACK_TIMEOUT_S,
)
from logger import current_session


# Safety filters off so design-related content isn't blocked.
_SAFETY_OFF = [
    genai.types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    genai.types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    genai.types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    genai.types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
]


def _build_client() -> genai.Client:
    key = get_gemini_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")
    return genai.Client(api_key=key)


def _call_sync(prompt: str, system_prompt: str, model: str) -> str:
    """Blocking Gemini call (runs in a thread via async wrapper)."""
    client = _build_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            safety_settings=_SAFETY_OFF,
        ),
    )
    if response.text is None:
        candidates = getattr(response, "candidates", None)
        reason = ""
        if candidates and len(candidates) > 0:
            reason = getattr(candidates[0], "finish_reason", "")
        raise RuntimeError(
            f"Gemini returned no text (finish_reason: {reason}). "
            "The model may not support this prompt."
        )
    return response.text


# Track which model last succeeded so routes can report it.
last_model_used: str = GEMINI_MODEL

# Sticky preferred model. Starts as the configured primary; whenever a call
# succeeds with a different model (because the preferred one failed), this is
# updated so future calls skip the broken model entirely.
_preferred_model: str = GEMINI_MODEL


def _build_try_order(start_model: str) -> list[str]:
    """Preferred model first, then the others in their configured order."""
    all_models = [GEMINI_MODEL, GEMINI_MODEL_FALLBACK, GEMINI_MODEL_FALLBACK_2]
    return [start_model] + [m for m in all_models if m != start_model]


async def generate(
    prompt: str,
    system_prompt: str,
    model: str | None = None,
    purpose: str = "text",
) -> str:
    """Async wrapper. Starts with the sticky preferred model (last success).
    Falls back through the others only if needed. Once any model succeeds,
    it becomes preferred so subsequent calls skip the broken ones.

    `purpose` is recorded in the per-request log (e.g. "stimuli_generation",
    "paired_prompts", "describe_image").
    """
    global last_model_used, _preferred_model

    start_model = model if model else _preferred_model
    models_to_try = _build_try_order(start_model)
    errors = []
    sess = current_session()

    for i, current_model in enumerate(models_to_try):
        timeout = API_TIMEOUT_S if i == 0 else FALLBACK_TIMEOUT_S
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_call_sync, prompt, system_prompt, current_model),
                timeout=timeout,
            )
            if sess is not None:
                sess.log_gemini_call(
                    purpose=purpose, model=current_model,
                    duration_s=time.monotonic() - t0, success=True,
                    prompt_chars=len(prompt) + len(system_prompt or ""),
                    response_chars=len(result),
                )
            last_model_used = current_model
            if current_model != _preferred_model:
                print(f"[gemini_client] Preferred model: {_preferred_model} → {current_model} (sticky)")
                _preferred_model = current_model
            return result
        except asyncio.TimeoutError:
            error_msg = f"{current_model}: timed out after {int(timeout)}s"
        except Exception as exc:
            error_msg = f"{current_model}: {type(exc).__name__}: {str(exc)[:200]}"

        if sess is not None:
            sess.log_gemini_call(
                purpose=purpose, model=current_model,
                duration_s=time.monotonic() - t0, success=False,
                prompt_chars=len(prompt) + len(system_prompt or ""),
                error=error_msg,
            )
        errors.append(error_msg)
        if i < len(models_to_try) - 1:
            print(f"[gemini_client] {error_msg}, trying next model...")

    raise RuntimeError("All models failed.\n" + "\n".join(f"  - {e}" for e in errors))


_DESCRIBE_SYSTEM_PROMPT = (
    "Describe what is shown and what is going on in one neutral sentence.\n"
    "Plain, factual, no preamble."
)


def _describe_image_sync(image_bytes: bytes, mime_type: str, model: str) -> str:
    """Send an image to Gemini and return a one-line description."""
    client = _build_client()
    response = client.models.generate_content(
        model=model,
        contents=[genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
        config=genai.types.GenerateContentConfig(
            system_instruction=_DESCRIBE_SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            safety_settings=_SAFETY_OFF,
        ),
    )
    text = (response.text or "").strip().replace("\n", " ")
    if not text:
        raise RuntimeError("Gemini returned no description text.")
    return text


async def describe_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Async wrapper: describe an image. Uses the sticky preferred model first."""
    global _preferred_model

    models_to_try = _build_try_order(_preferred_model)
    errors = []
    sess = current_session()

    for i, current_model in enumerate(models_to_try):
        timeout = API_TIMEOUT_S if i == 0 else FALLBACK_TIMEOUT_S
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_describe_image_sync, image_bytes, mime_type, current_model),
                timeout=timeout,
            )
            if sess is not None:
                sess.log_gemini_call(
                    purpose="describe_image", model=current_model,
                    duration_s=time.monotonic() - t0, success=True,
                    prompt_chars=len(image_bytes), response_chars=len(result),
                )
            if current_model != _preferred_model:
                print(f"[gemini_client] Preferred model: {_preferred_model} → {current_model} (sticky)")
                _preferred_model = current_model
            return result
        except Exception as exc:
            error_msg = f"{current_model}: {type(exc).__name__}: {str(exc)[:200]}"
            if sess is not None:
                sess.log_gemini_call(
                    purpose="describe_image", model=current_model,
                    duration_s=time.monotonic() - t0, success=False,
                    prompt_chars=len(image_bytes), error=error_msg,
                )
            errors.append(error_msg)
            if i < len(models_to_try) - 1:
                print(f"[gemini_client] Image description {error_msg}, trying next model...")

    raise RuntimeError("All description models failed.\n" + "\n".join(f"  - {e}" for e in errors))
