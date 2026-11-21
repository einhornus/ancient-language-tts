"""Old English. MOCK engine: grapheme tables only, no stress."""

from __future__ import annotations

from . import _mock

GRAPHEMES = {
    "ēa": "æːɑ", "ēo": "eːo", "ea": "æɑ", "eo": "eo",
    "sc": "ʃ", "cg": "dʒ", "ċ": "tʃ", "ġ": "j", "ng": "ŋg",
    "þ": "θ", "ð": "θ",
    "ā": "ɑː", "ǣ": "æː", "ē": "eː", "ī": "iː", "ō": "oː", "ū": "uː", "ȳ": "yː",
    "a": "ɑ", "c": "k", "x": "ks",
}

TO_ICELANDIC = {
    "θ": "þ", "ð": "ð", "ʃ": "sj", "tʃ": "tsj", "dʒ": "dj",
    "ɑ": "a", "æ": "æ", "ɣ": "g", "x": "h", "w": "v", "ŋ": "n",
}

ENGINE = _mock.engine("ang", GRAPHEMES, {"is": TO_ICELANDIC})
