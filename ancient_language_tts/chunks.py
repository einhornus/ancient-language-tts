"""How much text one IPA call may see.

`MAX_CHARS` is that limit, in characters of the source passage. A passage
within the limit is one piece. A longer passage is cut at the sentence
boundary closest to the middle, and both sides are cut the same way until
each piece fits or is a single sentence.

A sentence ends at `". "`, `"? "`, or `"! "`. The punctuation stays with
the sentence. One sentence is left whole, even when it is longer than the
limit: there is no boundary left to cut on.

`respell(method="llm")` transcribes each piece and respells that piece before the next
one. The pieces are joined again.
"""

from __future__ import annotations

# Characters of source text one IPA call receives. A short paragraph.
MAX_CHARS = 1000

_ENDS = (". ", "? ", "! ")


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
