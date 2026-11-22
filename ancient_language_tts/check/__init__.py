"""Check generated audio by recognizing its phonemes with a local ZIPA model.

    from ancient_language_tts import check

    check.hear("out/lat-ecc.it.1.mp3", "lat-ecc")
    # ["b", "ɔ", "r", "ɛ", "a", "s", " ", "ɛ", "t", ...]   tokens of lat-ecc
    frames = check.recognize_phonemes(audio_bytes, source="grc")
    # [{"start_ms": 40, "end_ms": 60, "confidence": 0.93,
    #   "probas": [("p", 0.81), ("b", 0.12), ...]}, ...]   one per confident frame
    check.phonemes_from_frames(frames, source="grc")
    # [{"phoneme": "pʰ", "start_ms": 40, "end_ms": 120, "confidence": 0.93,
    #   "probas": [("p", 0.81), ("b", 0.12), ...]}, ...]
    check.recognize("out/eme.en.1.mp3")
    # the model's own IPA, not conditioned on a language
    check.compare(expected_tokens, frames).mismatch
    # 0.12: frames aligned to the expected phonemes, penalty per weight (mismatch.py)

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
from .mismatch import (
    Comparison,
    Mistake,
    MistakeGroup,
    align,
    attach_modifiers,
    collapse_frames,
    compare,
    distance,
    frame_distance,
    heard_tokens,
    mismatch,
    phoneme_weights,
    phonemes_only,
    prepare_frames,
)
from .model import PhonemeCTCModel, get_phoneme_model, load_tokens
from .phonemes import (
    DEFAULT_THRESHOLD,
    MODIFIERS,
    PAUSE_MS,
    ZIPA_TOKENS,
    Frame,
    Phoneme,
    frames_from_logits,
    glyph_table,
    hear,
    tokens_from_phonemes,
    phonemes_from_frames,
    phonemes_from_logits,
    recognize,
    recognize_phonemes,
)

__all__ = [
    "Comparison",
    "DEFAULT_THRESHOLD",
    "Frame",
    "align",
    "attach_modifiers",
    "collapse_frames",
    "compare",
    "distance",
    "frame_distance",
    "heard_tokens",
    "mismatch",
    "phoneme_weights",
    "phonemes_only",
    "prepare_frames",
    "Language",
    "MODIFIERS",
    "Mistake",
    "MistakeGroup",
    "PAUSE_MS",
    "Phoneme",
    "PhonemeCTCModel",
    "ZIPA_TOKENS",
    "frames_from_logits",
    "get_phoneme_model",
    "glyph_table",
    "hear",
    "language",
    "load_audio",
    "load_tokens",
    "model_tokens",
    "phonemes_from_frames",
    "phonemes_from_logits",
    "recognize",
    "recognize_phonemes",
    "tokens_from_phonemes",
]
