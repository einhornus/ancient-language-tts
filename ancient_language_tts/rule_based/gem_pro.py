"""Proto-Germanic. MOCK engine: grapheme tables only, no stress."""

from __future__ import annotations

from . import _mock

GRAPHEMES = {
    "ng": "ŋg", "nk": "ŋk",
    "þ": "θ", "đ": "ð", "ƀ": "β", "ǥ": "ɣ", "h": "x",
    "ą": "ɑ̃", "į": "ĩ", "ų": "ũ",
    "ā": "ɑː", "ē": "eː", "ī": "iː", "ō": "oː", "ū": "uː",
    "a": "ɑ",
}

TO_ICELANDIC = {
    "θ": "þ", "ð": "ð", "β": "v", "ɣ": "g", "x": "h", "ŋ": "n",
    "ɑ̃": "an", "ĩ": "in", "ũ": "un", "ɑ": "a", "w": "v",
}

ENGINE = _mock.engine("gem-pro", GRAPHEMES, {"is": TO_ICELANDIC})
