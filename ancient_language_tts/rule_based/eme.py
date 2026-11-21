"""Early Modern English. MOCK engine: grapheme tables only, no stress."""

from __future__ import annotations

from . import _mock

GRAPHEMES = {
    "tion": "sɪən", "igh": "əɪ", "ee": "iː", "ea": "eː", "oo": "uː",
    "ou": "əʊ", "ow": "əʊ", "ai": "ɛː", "ay": "ɛː",
    "th": "θ", "sh": "ʃ", "ch": "tʃ", "ng": "ŋ", "ph": "f", "gh": "",
    "a": "a", "e": "ɛ", "i": "ɪ", "o": "ɔ", "u": "ʊ", "y": "ɪ",
    "c": "k", "j": "dʒ", "x": "ks",
}

TO_ENGLISH = {
    "əɪ": "uy", "əʊ": "oh", "iː": "ee", "eː": "ay", "ɛː": "eh", "uː": "oo",
    "oː": "oh", "aː": "ah",
    "θ": "th", "ð": "th", "ʃ": "sh", "tʃ": "ch", "dʒ": "j", "ŋ": "ng",
    "x": "kh", "ç": "kh",
    "ɪ": "i", "ɛ": "e", "ɔ": "o", "ʊ": "u", "ə": "uh", "a": "a",
}

ENGINE = _mock.engine("eme", GRAPHEMES, {"en": TO_ENGLISH})
