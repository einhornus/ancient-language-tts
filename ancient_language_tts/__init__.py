"""Text-to-speech for ancient languages, through a modern voice.

Bring your own keys: OpenRouter for the LLM calls, ElevenLabs for the speech.
The pipeline is four functions, each usable on its own:

    import ancient_language_tts as alt

    pieces = alt.to_ipa("Arma virumque cano", "lat-ecc", method="llm", tier="low")
    # [{"text": "Arma virumque cano", "ipa": ["ˈ", "a", "r", ...]}]
    print(alt.ipa_tokens.render(alt.chunks.join_ipa(pieces)))
    line = alt.respell_ipa(pieces, "lat-ecc", "it", method="llm")
    line = alt.respell("Arma virumque cano", "lat-ecc", "it", method="rules")
    audio = alt.speak(line, "it", output="out/lat-ecc.mp3")

`method` is "llm" (the same text → IPA → respelling pipeline for every
language) or "rules" (the deterministic engine in `rule_based`).
"""

from . import chunks, ipa_tokens, llm, rule_based
from .api import METHODS, Method, respell, respell_ipa, speak, to_ipa
from .chunks import MAX_CHARS, Chunk, split
from .config import EFFORTS, TIER_MODELS, Effort, Settings, Tier
from .errors import TTSError
from .languages import ancient_languages, modern_languages
from .llm import improve_ipa, improve_respelling, score_ipa, score_respelling
from .rules import targets

__version__ = "0.2.0"

__all__ = [
    "Chunk",
    "EFFORTS",
    "Effort",
    "MAX_CHARS",
    "METHODS",
    "Method",
    "Settings",
    "TIER_MODELS",
    "TTSError",
    "Tier",
    "ancient_languages",
    "chunks",
    "improve_ipa",
    "ipa_tokens",
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
