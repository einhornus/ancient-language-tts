"""Middle English. MOCK engine: grapheme tables only, no stress."""

from __future__ import annotations

from . import _mock

GRAPHEMES = {
    "gh": "ç", "ee": "eː", "oo": "oː", "ou": "uː", "ow": "uː",
    "ch": "tʃ", "sh": "ʃ", "th": "θ", "ng": "ŋg", "þ": "θ", "ȝ": "j",
    "a": "a", "e": "ɛ", "i": "ɪ", "o": "ɔ", "u": "ʊ", "y": "ɪ",
    "c": "k", "j": "dʒ", "x": "ks",
}

TO_ENGLISH = {
    "eː": "ay", "oː": "oh", "uː": "oo", "iː": "ee", "aː": "ah",
    "ç": "kh", "x": "kh", "tʃ": "ch", "ʃ": "sh", "θ": "th", "ð": "th",
    "dʒ": "j", "ŋ": "n",
    "ɪ": "i", "ɛ": "e", "ɔ": "o", "ʊ": "u", "ə": "uh", "a": "ah",
}

TO_DUTCH = {
    "eː": "ee", "oː": "oo", "uː": "oe", "iː": "ie", "aː": "aa",
    "ç": "ch", "x": "ch", "tʃ": "tsj", "ʃ": "sj", "θ": "t", "ð": "d",
    "dʒ": "dzj", "ŋ": "n",
    "ɪ": "i", "ɛ": "e", "ɔ": "o", "ʊ": "oe", "ə": "e", "a": "a",
}

ENGINE = _mock.engine("enm", GRAPHEMES, {"en": TO_ENGLISH, "nl": TO_DUTCH})
