"""Text-to-speech for ancient languages, through a modern voice.

Bring your own keys: OpenRouter for the LLM calls, ElevenLabs for the speech.
The pipeline is four functions, each usable on its own:

    import ancient_language_tts as alt

    ipa = alt.to_ipa("Arma virumque canō", "lat", method="llm", tier="low")
    line = alt.respell_ipa(ipa, "lat", "it", method="llm")
    line = alt.respell("Arma virumque canō", "lat", "it", method="rules")
    audio = alt.speak(line, "it", output="out/lat.mp3")

`method` is "llm" (the same text → IPA → respelling pipeline for every
language) or "rules" (the deterministic engine in `rule_based`).
"""

from . import llm, rule_based
from .api import METHODS, Method, respell, respell_ipa, speak, to_ipa
from .chunks import MAX_CHARS, split
from .config import TIER_MODELS, Settings, Tier
from .errors import TTSError
from .languages import ancient_languages, modern_languages
from .llm import improve_ipa, improve_respelling, score_ipa, score_respelling
from .rules import targets

__version__ = "0.2.0"

__all__ = [
    "MAX_CHARS",
    "METHODS",
    "Method",
    "Settings",
    "TIER_MODELS",
    "TTSError",
    "Tier",
    "ancient_languages",
    "improve_ipa",
    "improve_respelling",
    "llm",
    "modern_languages",
    "respell",
    "respell_ipa",
    "rule_based",
    "score_ipa",
    "score_respelling",
    "speak",
    "split",
    "targets",
    "to_ipa",
]
