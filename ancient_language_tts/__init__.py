"""Text-to-speech for ancient languages, through a modern voice.

Bring your own keys: OpenRouter for the LLM calls, ElevenLabs for the speech.

    import ancient_language_tts

    speech = ancient_language_tts.speak(
        "Arma virumque canō",
        source="lat",
        target="en",
        output="out/lat.mp3",
        openrouter_api_key="sk-or-...",
        elevenlabs_api_key="...",
    )
"""

from . import rule_based
from .chunks import MAX_CHARS, split
from .config import TIER_MODELS, Settings, Tier
from .errors import TTSError
from .ipa import improve_ipa, score_ipa, to_ipa
from .languages import ancient_languages, modern_languages
from .respelling import improve_respelling, respell, score_respelling
from .rules import targets
from .speak import Speech, speak
from .tts import synthesize

__version__ = "0.1.0"

__all__ = [
    "MAX_CHARS",
    "Speech",
    "Settings",
    "TTSError",
    "TIER_MODELS",
    "Tier",
    "ancient_languages",
    "improve_ipa",
    "improve_respelling",
    "modern_languages",
    "respell",
    "rule_based",
    "score_ipa",
    "score_respelling",
    "speak",
    "split",
    "synthesize",
    "targets",
    "to_ipa",
]
