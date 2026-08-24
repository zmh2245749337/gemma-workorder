"""Local environment helpers: `.env` loading and Hugging Face token resolution.

Colab supplies ``HF_TOKEN`` through Colab Secrets, which are injected into the
process environment before any script runs. Running the same scripts locally
needs an equivalent that does not require re-exporting an environment
variable in every new terminal, so this module adds one extra, backward
compatible fallback: a local ``.env`` file (never committed — see
``.gitignore``) and, failing that, a token cached by ``huggingface-cli
login``. Nothing here changes behavior on Colab, where ``HF_TOKEN`` is
already present in the environment.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is an optional convenience; env vars set any other way
    # (Colab Secrets, `export HF_TOKEN=...`, CI secrets) still work without it.
    pass


def resolve_hf_token() -> str | bool:
    """Return the Hugging Face token to use for gated model access.

    Resolution order:
    1. ``HF_TOKEN`` from the process environment or a local ``.env`` file.
    2. ``True``, which tells ``huggingface_hub`` to use the token cached by
       a prior ``huggingface-cli login`` (or ``hf auth login``) run.

    Returns ``True`` rather than ``None`` in the fallback case so a one-time
    local login keeps working even when no ``HF_TOKEN`` is set anywhere.
    """
    token = os.getenv("HF_TOKEN")
    return token if token else True
