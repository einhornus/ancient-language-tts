"""Check generated audio by recognizing its phonemes with a local ZIPA model.

    from ancient_language_tts import check

    check.hear("out/lat-ecc.it.1.mp3", "lat-ecc")
    # ["b", "ɔ", "r", "ɛ", "a", "s", " ", "ɛ", "t", ...]   tokens of lat-ecc
    check.recognize_phonemes(audio_bytes, source="grc")
    # [{"phoneme": "pʰ", "start_ms": 40, "end_ms": 120, "confidence": 0.93,
    #   "probas": [("pʰ", 0.81), ("p", 0.12), ...]}, ...]
    check.recognize("out/eme.en.1.mp3")
    # the model's own IPA, not conditioned on a language
    check.mismatch(expected_tokens, heard_tokens)
    # 0.12: weighted phoneme edit distance per expected phoneme (mismatch.py)

ZIPA is a multilingual phone recognizer (88 training languages) that writes
IPA directly (its vocabulary: `data/zipa_tokens.json`). With a `source`,
recognition is conditioned on that language through
`data/<source>/zipa.json`; see `conditioning.py`. Needs the optional
dependencies: pip install 'ancient-language-tts[check]'. The ONNX weights
(~1.2 GB, or `ZIPA_MODEL_FILE=model.int8.onnx` for ~300 MB) download from
the Hugging Face Hub on first use.
"""

from .audio import load_audio
from .conditioning import Language, language, model_tokens
from .mismatch import Comparison, Mistake, compare, mismatch, phonemes_only, substitution_cost
from .model import PhonemeCTCModel, get_phoneme_model, load_tokens
from .phonemes import (
    DEFAULT_THRESHOLD,
    MODIFIERS,
    PAUSE_MS,
    ZIPA_TOKENS,
    Phoneme,
    glyph_table,
    hear,
    tokens_from_phonemes,
    phonemes_from_logits,
    recognize,
    recognize_phonemes,
)

__all__ = [
    "Comparison",
    "DEFAULT_THRESHOLD",
    "compare",
    "mismatch",
    "phonemes_only",
    "substitution_cost",
    "Language",
    "MODIFIERS",
    "Mistake",
    "PAUSE_MS",
    "Phoneme",
    "PhonemeCTCModel",
    "ZIPA_TOKENS",
    "get_phoneme_model",
    "glyph_table",
    "hear",
    "language",
    "load_audio",
    "load_tokens",
    "model_tokens",
    "phonemes_from_logits",
    "recognize",
    "recognize_phonemes",
    "tokens_from_phonemes",
]
