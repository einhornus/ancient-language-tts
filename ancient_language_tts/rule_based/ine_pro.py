"""Proto-Indo-European. MOCK engine: grapheme tables only, no stress."""

from __future__ import annotations

from . import _mock

GRAPHEMES = {
    "h₁": "h", "h₂": "χ", "h₃": "ɣʷ", "hₓ": "h",
    "ǵʰ": "ɟʱ", "gʷʰ": "gʷʱ", "bʰ": "bʱ", "dʰ": "dʱ", "gʰ": "gʱ",
    "ḱ": "c", "ǵ": "ɟ",
    "r̥": "r̩", "l̥": "l̩", "m̥": "m̩", "n̥": "n̩",
    "ā": "aː", "ē": "eː", "ī": "iː", "ō": "oː", "ū": "uː",
    "y": "j",
}

TO_LITHUANIAN = {
    "χ": "ch", "ɣʷ": "gv", "ɟʱ": "gj", "gʷʱ": "gv", "gʷ": "gv", "kʷ": "kv",
    "bʱ": "b", "dʱ": "d", "gʱ": "g", "c": "kj", "ɟ": "gj",
    "r̩": "ir", "l̩": "il", "m̩": "im", "n̩": "in",
    "eː": "ė", "oː": "o", "aː": "a", "iː": "y", "uː": "ū",
    "ə": "e", "w": "v",
}

ENGINE = _mock.engine("ine-pro", GRAPHEMES, {"lt": TO_LITHUANIAN})
