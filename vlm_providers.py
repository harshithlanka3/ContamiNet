"""VLM backends: local Ollama vs Google AI Studio (Gemini)."""

from __future__ import annotations

import asyncio
import os
from typing import Literal

Provider = Literal["ollama", "google"]


def parse_vlm_provider() -> Provider:
    raw = os.environ.get("CONTAMINET_VLM_PROVIDER", "ollama").strip().lower()
    if raw in ("ollama", "local"):
        return "ollama"
    if raw in ("google", "gemini", "google_ai", "aistudio", "google-ai-studio"):
        return "google"
    raise ValueError(
        f"Invalid CONTAMINET_VLM_PROVIDER={raw!r}; use 'ollama' or 'google'"
    )


def google_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Google VLM requires GEMINI_API_KEY or GOOGLE_API_KEY in the environment"
        )
    return key


def init_google_backend() -> None:
    """Call once at process startup when using the Google provider."""
    import google.generativeai as genai

    genai.configure(api_key=google_api_key())


async def vlm_generate_json_text(
    provider: Provider,
    *,
    prompt: str,
    image_bytes: bytes,
    image_mime: str,
    ollama_host: str,
    ollama_model: str,
    gemini_model: str,
) -> str:
    """
    Send prompt + image to the configured VLM and return raw text (expected JSON).
    """
    if provider == "ollama":
        from ollama import AsyncClient

        client = AsyncClient(host=ollama_host)
        response = await client.chat(
            model=ollama_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }
            ],
            format="json",
        )
        return response["message"]["content"]

    if provider != "google":
        raise ValueError(f"Unknown VLM provider: {provider!r}")

    def _run_google() -> str:
        import google.generativeai as genai

        model = genai.GenerativeModel(
            gemini_model,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            ),
        )
        response = model.generate_content(
            [
                prompt,
                {"mime_type": image_mime, "data": image_bytes},
            ],
        )
        if not response.candidates:
            fb = getattr(response, "prompt_feedback", None)
            raise RuntimeError(f"Gemini returned no candidates (prompt_feedback={fb!r})")
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty text")
        return text

    return await asyncio.to_thread(_run_google)
