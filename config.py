"""
Central configuration for the Design Stimulus Generator.
Single source of truth for model, paths, and environment.
"""

import os
import re
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
ENV_PATH = BASE_DIR / ".env"

IMAGE_BACKUP_DIR = BASE_DIR / "image_backups"
IMAGE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_IMAGES_DIR = BASE_DIR / "generated_images"
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADED_IMAGES_DIR = BASE_DIR / "uploaded_images"
UPLOADED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ───────────────────────────────────────────────────────────────────
# Text generation (Gemini prompts) — flash is primary for now (faster).
GEMINI_MODEL = "gemini-3.1-pro-preview"
GEMINI_MODEL_FALLBACK = "gemini-3-flash-preview" 
GEMINI_MODEL_FALLBACK_2 = "gemini-2.5-pro"

# Image generation (Imagen)
IMAGEN_MODEL = "gemini-3-pro-image-preview"
IMAGEN_MODEL_FALLBACK = "gemini-3.1-flash-image-preview"
IMAGEN_MODEL_FALLBACK_2 = "gemini-2.5-flash-image"

FALLBACK_TIMEOUT_S = 120.0

# ── Generation settings ──────────────────────────────────────────────────────
MAX_OUTPUT_TOKENS = 32768
API_TIMEOUT_S = float(os.getenv("API_TIMEOUT_S", "180"))
EXPECTED_SENTENCES = 5

# ── .env loader ──────────────────────────────────────────────────────────────
_env_cache: dict[str, str] | None = None


def _load_env() -> dict[str, str]:
    global _env_cache
    if _env_cache is not None:
        return _env_cache
    result: dict[str, str] = {}
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip().strip('"').strip("'")
    _env_cache = result
    return result


def get_gemini_key() -> str | None:
    return _load_env().get("GEMINI_API_KEY")


# ── Naming helpers ─────────────────────────────────────────────────────────

def task_slug(task: str, max_len: int = 40) -> str:
    """Convert a task description into a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", task.strip()).strip("_").lower()
    return (slug or "task")[:max_len]


def next_available_name(directory: Path, base_name: str, extension: str = "") -> str:
    """Return the next available name with incremental numbering.

    First occurrence:  {base_name}{extension}
    Duplicates:        {base_name}_1{extension}, {base_name}_2{extension}, ...
    """
    candidate = f"{base_name}{extension}"
    if not (directory / candidate).exists():
        return candidate
    n = 1
    while True:
        candidate = f"{base_name}_{n}{extension}"
        if not (directory / candidate).exists():
            return candidate
        n += 1


