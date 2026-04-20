"""VLM backends: local Ollama vs Google AI Studio (Gemini via google-genai SDK)."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

Provider = Literal["ollama", "google"]

# Set in init_google_backend() when CONTAMINET_VLM_PROVIDER=google
_google_genai_client: Any = None


class GeminiQuotaOrRateLimit(Exception):
    """Gemini API returned 429 or quota / rate-limit exhaustion."""

    pass


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
    global _google_genai_client
    from google import genai

    _google_genai_client = genai.Client(api_key=google_api_key())


def _gemini_client() -> Any:
    global _google_genai_client
    if _google_genai_client is None:
        from google import genai

        _google_genai_client = genai.Client(api_key=google_api_key())
    return _google_genai_client


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
        from google.genai import errors as genai_errors
        from google.genai import types

        client = _gemini_client()
        try:
            response = client.models.generate_content(
                model=gemini_model,
                contents=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
        except genai_errors.APIError as e:
            if getattr(e, "code", None) == 429:
                raise GeminiQuotaOrRateLimit(str(e)) from e
            raise
        except Exception as e:
            msg = str(e)
            if "429" in msg or (
                "quota" in msg.lower() and "exceed" in msg.lower()
            ):
                raise GeminiQuotaOrRateLimit(msg) from e
            raise

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty text")
        return text

    return await asyncio.to_thread(_run_google)
