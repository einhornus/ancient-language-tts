"""LLM-based IPA transcription and respelling."""

from .client import ask, require_key
from .ipa import improve_ipa, score_ipa, to_ipa
from .respelling import improve_respelling, respell, score_respelling

__all__ = [
    "ask",
    "improve_ipa",
    "improve_respelling",
    "require_key",
    "respell",
    "score_ipa",
    "score_respelling",
    "to_ipa",
]
