"""Deterministic IPA and respelling, when a pair has a rule engine.

Rule modules live under this package. LLM-based steps live under
`ancient_language_tts.llm`. Call `available` / `has` to see which pairs
are implemented here.
"""

from __future__ import annotations

from typing import Callable

from . import lat_ecc_it

# (source, target) → (to_ipa, respell_from_ipa, latin_to_italian)
_PAIRS: dict[tuple[str, str], tuple[Callable[[str], str], Callable[[str], str], Callable[[str], str]]] = {
    (lat_ecc_it.SOURCE, lat_ecc_it.TARGET): (
        lat_ecc_it.to_ipa,
        lat_ecc_it.respell,
        lat_ecc_it.to_italian,
    ),
}


def available() -> list[tuple[str, str]]:
    """Language pairs that have a rule-based pipeline."""
    return sorted(_PAIRS)


def has(source: str, target: str) -> bool:
    """True when `(source, target)` has a rule-based IPA + respelling path."""
    from .. import languages

    src = languages.ancient(source).code
    tgt = languages.modern(target).code
    return (src, tgt) in _PAIRS


def to_ipa(text: str, source: str, target: str) -> str:
    """Rule-based IPA for `text`. Raises if the pair has no rule engine."""
    ipa_fn, _, _ = _pair(source, target)
    return ipa_fn(text)


def respell(ipa: str, source: str, target: str) -> str:
    """Rule-based respelling of `ipa`. Raises if the pair has no rule engine."""
    _, respell_fn, _ = _pair(source, target)
    return respell_fn(ipa)


def direct(text: str, source: str, target: str) -> str:
    """Latin spelling → modern spelling, with no IPA step."""
    _, _, direct_fn = _pair(source, target)
    return direct_fn(text)


def convert(text: str, source: str, target: str) -> tuple[str, str]:
    """Return `(ipa, respelling)` using the rule engine for this pair."""
    ipa_fn, respell_fn, _ = _pair(source, target)
    ipa = ipa_fn(text)
    return ipa, respell_fn(ipa)


def _pair(
    source: str, target: str
) -> tuple[Callable[[str], str], Callable[[str], str], Callable[[str], str]]:
    from .. import languages

    src = languages.ancient(source).code
    tgt = languages.modern(target).code
    try:
        return _PAIRS[(src, tgt)]
    except KeyError:
        have = ", ".join(f"{s}→{t}" for s, t in available()) or "(none)"
        raise ValueError(
            f"no rule-based solution for {src} → {tgt}. Available: {have}."
        ) from None


__all__ = [
    "available",
    "convert",
    "direct",
    "has",
    "respell",
    "to_ipa",
]
