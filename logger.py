"""
Comprehensive per-request backend logger.

Each frontend request opens a GenerationSession (via session_scope) which is
exposed to the rest of the backend through contextvars. Every Gemini/Imagen
call records its model, timing, and outcome into the active session. When the
request finishes (success or error), one JSON line is appended to
generation_log.jsonl containing the full trace:

  - frontend payload (task, requirements, format, endpoint)
  - generated stimuli (NC / MC / VC)
  - paired Imagen prompts per sentence position
  - every model call (text + image) with model name + duration + success
  - final image filenames and sizes
  - total pipeline duration
"""

import contextvars
import json
import threading
import time
from datetime import datetime
from typing import Any

from config import BASE_DIR

LOG_PATH = BASE_DIR / "generation_log.jsonl"
_write_lock = threading.Lock()
_session_var: contextvars.ContextVar = contextvars.ContextVar(
    "generation_session", default=None
)


class GenerationSession:
    def __init__(self, endpoint: str, request_payload: dict[str, Any]):
        self.session_id   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.started_at   = datetime.now()
        self._t0          = time.monotonic()
        self.endpoint     = endpoint
        self.frontend     = request_payload
        self.gemini_calls: list[dict] = []
        self.image_calls:  list[dict] = []
        self.paired_prompts: list[dict] = []
        self.stimuli: dict | None    = None
        self.images: dict | None     = None
        self.descriptions: list[dict] = []
        self.notes: list[dict]       = []

    # ── per-call recorders (called from clients) ──────────────────────────────

    def log_gemini_call(self, *, purpose: str, model: str, duration_s: float,
                        success: bool, prompt_chars: int = 0,
                        response_chars: int = 0, error: str | None = None) -> None:
        self.gemini_calls.append({
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "purpose":        purpose,
            "model":          model,
            "duration_s":     round(duration_s, 3),
            "prompt_chars":   prompt_chars,
            "response_chars": response_chars,
            "success":        success,
            "error":          error,
        })

    def log_image_call(self, *, position: str, level: str, model: str,
                       duration_s: float, success: bool, prompt_chars: int = 0,
                       filename: str | None = None, size_bytes: int | None = None,
                       error: str | None = None) -> None:
        self.image_calls.append({
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
            "position":    position,
            "level":       level,
            "model":       model,
            "duration_s":  round(duration_s, 3),
            "prompt_chars": prompt_chars,
            "filename":    filename,
            "size_bytes":  size_bytes,
            "success":     success,
            "error":       error,
        })

    def log_paired_prompts(self, *, position: str, nc_sentence: str, vc_sentence: str,
                           nc_prompt: str, vc_prompt: str) -> None:
        self.paired_prompts.append({
            "position":     position,
            "nc_sentence":  nc_sentence,
            "vc_sentence":  vc_sentence,
            "nc_prompt":    nc_prompt,
            "vc_prompt":    vc_prompt,
        })

    def log_description(self, *, position: str, filename: str,
                        description: str, model: str | None = None) -> None:
        self.descriptions.append({
            "position":    position,
            "filename":    filename,
            "description": description,
            "model":       model,
        })

    # ── high-level setters (called once from routes) ──────────────────────────

    def set_stimuli(self, stimuli: dict[str, str], report_file: str | None = None) -> None:
        self.stimuli = {**stimuli, "report_file": report_file}

    def set_images(self, summary: dict[str, Any]) -> None:
        self.images = summary

    def note(self, message: str, **extra) -> None:
        self.notes.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "message":   message,
            **extra,
        })

    # ── finalize ──────────────────────────────────────────────────────────────

    def finalize(self, status: str, error: str | None = None) -> dict:
        duration_s = round(time.monotonic() - self._t0, 3)
        entry = {
            "session_id":   self.session_id,
            "endpoint":     self.endpoint,
            "started_at":   self.started_at.isoformat(timespec="seconds"),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "duration_s":   duration_s,
            "status":       status,
            "error":        error,
            "frontend_request": self.frontend,
            "stimuli":           self.stimuli,
            "paired_prompts":    self.paired_prompts,
            "images":            self.images,
            "descriptions":      self.descriptions or None,
            "gemini_calls":      self.gemini_calls,
            "image_calls":       self.image_calls,
            "notes":             self.notes or None,
            "totals": {
                "gemini_calls":   len(self.gemini_calls),
                "image_calls":    len(self.image_calls),
                "gemini_total_s": round(sum(c["duration_s"] for c in self.gemini_calls), 3),
                "image_total_s":  round(sum(c["duration_s"] for c in self.image_calls), 3),
            },
        }
        with _write_lock:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry


def current_session() -> GenerationSession | None:
    """Return the GenerationSession for the active request, or None."""
    return _session_var.get()


class session_scope:
    """Context manager opening a GenerationSession bound to contextvars.

    Usable as a sync `with` block. Works correctly inside async handlers and
    is propagated to threads spawned via asyncio.to_thread (Python contextvars
    are copied per-task and per-to_thread call automatically).
    """

    def __init__(self, endpoint: str, request_payload: dict[str, Any]):
        self.endpoint = endpoint
        self.payload  = request_payload
        self.session: GenerationSession | None = None
        self._token = None

    def __enter__(self) -> GenerationSession:
        self.session = GenerationSession(self.endpoint, self.payload)
        self._token  = _session_var.set(self.session)
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self.session.finalize("success")
            else:
                err = f"{exc_type.__name__}: {str(exc_val)[:500]}"
                self.session.finalize("error", error=err)
        finally:
            if self._token is not None:
                _session_var.reset(self._token)
        return False
