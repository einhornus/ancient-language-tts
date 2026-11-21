"""Deterministic IPA and respelling, one engine per ancient language.

Every ancient language has an engine with three conversions (see `base`):
text → IPA, text → respelling, IPA → respelling. Only Ecclesiastical Latin
(`lat-ecc`, into Italian) has real rules today; the others are mocks built
from small grapheme tables (`is_mock` tells them apart). LLM-based steps
live under `ancient_language_tts.llm`.
"""

from __future__ import annotations

from .. import languages
from . import ang, eme, enm, gem_pro, grc, ine_pro, lat, lat_ecc, sla_pro
from .base import Convert, RuleEngine

_ENGINES: dict[str, RuleEngine] = {
    module.ENGINE.source: module.ENGINE
    for module in (lat, lat_ecc, grc, eme, enm, ang, gem_pro, ine_pro, sla_pro)
}


def engine(source: str) -> RuleEngine:
    """The rule engine for one ancient language."""
    code = languages.ancient(source).code
    try:
        return _ENGINES[code]
    except KeyError:
        have = ", ".join(sorted(_ENGINES))
        raise ValueError(f"no rule engine for {code}. Engines: {have}.") from None


def available() -> list[tuple[str, str]]:
    """Every `(source, target)` pair a rule engine can respell."""
    return sorted(
        (code, target) for code, item in _ENGINES.items() for target in item.targets()
    )


def has(source: str, target: str) -> bool:
    """True when `source` has rule-based respelling into `target`."""
    return languages.modern(target).code in engine(source).targets()


def is_mock(source: str) -> bool:
    """True when the engine for `source` is a placeholder, not real rules."""
    return engine(source).mock


def to_ipa(text: str, source: str) -> str:
    """Rule-based IPA for `text`. Does not depend on a modern language."""
    return engine(source).to_ipa(text)


def respell(text: str, source: str, target: str) -> str:
    """Rule-based respelling of the source spelling, with no IPA step."""
    return _pick(engine(source), "respell", target)(text)


def respell_ipa(ipa: str, source: str, target: str) -> str:
    """Rule-based respelling of an IPA line."""
    return _pick(engine(source), "respell_ipa", target)(ipa)


def _pick(item: RuleEngine, kind: str, target: str) -> Convert:
    code = languages.modern(target).code
    table = getattr(item, kind)
    try:
        return table[code]
    except KeyError:
        have = ", ".join(sorted(table)) or "(none)"
        raise ValueError(
            f"no rule-based {kind} for {item.source} → {code}. "
            f"Targets for {item.source}: {have}."
        ) from None


__all__ = [
    "RuleEngine",
    "available",
    "engine",
    "has",
    "is_mock",
    "respell",
    "respell_ipa",
    "to_ipa",
]
