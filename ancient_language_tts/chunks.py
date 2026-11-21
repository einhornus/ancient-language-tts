"""How much text one IPA call may see.

`MAX_CHARS` is that limit, in characters of the source passage. A passage
within the limit is one piece. A longer passage is cut at the sentence
boundary closest to the middle, and both sides are cut the same way until
each piece fits or is a single sentence.

A sentence ends at `". "`, `"? "`, or `"! "`. The punctuation stays with
the sentence. One sentence is left whole, even when it is longer than the
limit: there is no boundary left to cut on.

`to_ipa` returns its pieces as `Chunk`s: a dict with the piece's `text`
and its `ipa` tokens. The respelling step takes those chunks, so each piece
of IPA is respelled with exactly the text it came from.
`respell(method="llm")` transcribes each piece and respells that piece
before the next one. The pieces are joined again.

`split_ipa` cuts a bare IPA line, one with no chunks, the same way. A
sentence of the line ends at a ".", "?", "!", ";" or ":" token followed by
a space token, and a sentence of the original text at the same marks
followed by a space. When both have the same number of sentences, the
k-th IPA sentence goes with the k-th sentence of the text, and the
lengths of the text decide the cuts, as in `split`. Otherwise the IPA is
cut on its own rendered length and the pieces carry no text.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

# Characters of source text one IPA call receives. A short paragraph.
MAX_CHARS = 1000

_ENDS = (". ", "? ", "! ")

# Marks that end a sentence for `split_ipa`. The IPA step writes a period
# for a semicolon or a colon, and the rule engines keep them, so both
# sides count all five.
_PAIRED_ENDS = (".", "?", "!", ";", ":")
# The text side also has the Greek ano teleia (a semicolon), in both its
# code points, and the Greek question mark.
_TEXT_ENDS = _PAIRED_ENDS + ("·", "·", ";")
_SPACE = " "


class Chunk(TypedDict):
    """One piece of a passage: its source `text` and the IPA tokens of that text."""

    text: str
    ipa: list[str]


def join_ipa(pieces: Sequence[Chunk]) -> list[str]:
    """The IPA tokens of every chunk, in order, with a space token between chunks."""
    tokens: list[str] = []
    for piece in pieces:
        if tokens:
            tokens.append(_SPACE)
        tokens.extend(piece["ipa"])
    return tokens


def join_text(pieces: Sequence[Chunk]) -> str:
    """The source text of every chunk, in order, joined with a space."""
    return " ".join(piece["text"] for piece in pieces if piece["text"])


def from_ipa(ipa: Sequence[str], text: str = "", limit: int | None = None) -> tuple[list[Chunk], bool]:
    """Chunks for a bare IPA token list, cut as `split_ipa` cuts it.

    Returns the chunks and whether `text` could be paired with the IPA.
    When it could not, every chunk's text is "".
    """
    pieces, aligned = split_ipa(ipa, text, limit)
    return [{"text": part_text, "ipa": part} for part, part_text in pieces], aligned


def ensure(pieces: Any) -> list[Chunk]:
    """Check that `pieces` is a list of chunks, as `to_ipa` returns it, and copy it.

    A bare token list or a string is refused: use `from_ipa` for a token list.
    """
    message = "expected a list of chunks ({'text': str, 'ipa': list[str]}), as to_ipa returns it"
    if isinstance(pieces, (str, dict)) or not isinstance(pieces, Sequence):
        raise TypeError(message)
    out: list[Chunk] = []
    for piece in pieces:
        if not isinstance(piece, dict) or "ipa" not in piece:
            raise TypeError(message)
        out.append({"text": str(piece.get("text") or ""), "ipa": list(piece["ipa"])})
    if not out:
        raise ValueError("no chunks")
    return out


def split(text: str, limit: int | None = None) -> list[str]:
    """Split `text` into pieces no longer than `limit`.

    `limit` defaults to the run's `max_chars` setting, which defaults to
    `MAX_CHARS`. An empty passage returns an empty list. A passage the
    limit already holds, and a passage of one sentence, returns that
    passage as the only piece.
    """
    if limit is None:
        from .config import current

        limit = current().max_chars
    if limit < 1:
        raise ValueError("max length must be at least 1")
    cleaned = text.strip()
    if not cleaned:
        return []
    return _split(cleaned, limit)


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    cut = _nearest_cut(text)
    if cut is None:
        return [text]
    left = text[:cut].strip()
    right = text[cut:].strip()
    if not left or not right:
        return [text]
    return _split(left, limit) + _split(right, limit)


def _nearest_cut(text: str) -> int | None:
    """Index where the next sentence starts, at the boundary closest to the middle.

    The index is the first character of the following sentence. A boundary
    that would leave either side empty is ignored. When two boundaries are
    the same distance from the middle, the earlier one is used. `None`
    means the passage is a single sentence.
    """
    cuts: list[int] = []
    index = 0
    length = len(text)
    while index < length - 1:
        if text[index:index + 2] in _ENDS:
            cuts.append(index + 2)
            index += 2
        else:
            index += 1
    usable = [cut for cut in cuts if text[:cut].strip() and text[cut:].strip()]
    if not usable:
        return None
    middle = length / 2
    return min(usable, key=lambda cut: (abs(cut - middle), cut))


def split_ipa(
    ipa: Sequence[str], text: str = "", limit: int | None = None
) -> tuple[list[tuple[list[str], str]], bool]:
    """Split IPA tokens, and the text they came from, into pieces of at most `limit`.

    Returns the `(tokens, text)` pieces and whether the text could be
    paired with the IPA. When it could not, each piece's text is "". The
    limit defaults to the run's `max_chars` setting. Empty IPA returns no
    pieces.
    """
    if limit is None:
        from .config import current

        limit = current().max_chars
    if limit < 1:
        raise ValueError("max length must be at least 1")
    tokens = list(ipa)
    spans = _ipa_sentences(tokens)
    if not spans:
        return [], bool(text.strip())
    text_spans = _text_sentences(text)
    aligned = bool(text_spans) and len(text_spans) == len(spans)
    if aligned:
        lengths = [end - start for start, end in text_spans]
    else:
        lengths = [len("".join(tokens[start:end])) for start, end in spans]
    pieces = []
    for first, last in _group(lengths, 0, len(lengths), limit):
        part = _strip(tokens[spans[first][0]:spans[last - 1][1]])
        part_text = ""
        if aligned:
            part_text = text[text_spans[first][0]:text_spans[last - 1][1]].strip()
        pieces.append((part, part_text))
    return pieces, aligned


def _strip(tokens: list[str]) -> list[str]:
    start, end = 0, len(tokens)
    while start < end and not tokens[start].strip():
        start += 1
    while end > start and not tokens[end - 1].strip():
        end -= 1
    return tokens[start:end]


def _ipa_sentences(tokens: list[str]) -> list[tuple[int, int]]:
    """Token ranges of the sentences, each ending after its closing mark."""
    spans, start = [], 0
    for index in range(len(tokens) - 1):
        if tokens[index] in _PAIRED_ENDS and tokens[index + 1] == _SPACE:
            if _strip(tokens[start:index + 1]):
                spans.append((start, index + 1))
                start = index + 1
    if _strip(tokens[start:]):
        spans.append((start, len(tokens)))
    elif spans:
        spans[-1] = (spans[-1][0], len(tokens))
    return spans


def _text_sentences(text: str) -> list[tuple[int, int]]:
    """Character ranges of the sentences, each ending after its closing mark."""
    spans, start = [], 0
    for index in range(len(text) - 1):
        if text[index] in _TEXT_ENDS and text[index + 1] == _SPACE:
            if text[start:index + 1].strip():
                spans.append((start, index + 1))
                start = index + 1
    if text[start:].strip():
        spans.append((start, len(text)))
    elif spans:
        spans[-1] = (spans[-1][0], len(text))
    return spans


def _group(lengths: list[int], first: int, last: int, limit: int) -> list[tuple[int, int]]:
    """Cut sentences first..last-1 like `_split`: at the boundary closest to the middle."""
    size = sum(lengths[first:last]) + (last - first - 1)
    if size <= limit or last - first <= 1:
        return [(first, last)]
    middle = size / 2
    best, best_distance, before = first + 1, None, 0
    for cut in range(first + 1, last):
        before += lengths[cut - 1] + (1 if cut - 1 > first else 0)
        distance = abs(before - middle)
        if best_distance is None or distance < best_distance:
            best, best_distance = cut, distance
    return _group(lengths, first, best, limit) + _group(lengths, best, last, limit)
