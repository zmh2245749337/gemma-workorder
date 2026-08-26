"""Local `.env` loading and Hugging Face token resolution."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Environment variables and cached Hub credentials still work without it.
    pass


def resolve_hf_token() -> str | bool:
    """Use ``HF_TOKEN`` when set, otherwise the credential cached by Hub login."""

    token = os.getenv("HF_TOKEN")
    return token if token else True
