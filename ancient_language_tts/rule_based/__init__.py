"""Deterministic IPA and respelling, one engine per ancient language.

Every ancient language has an engine with three conversions (see `base`):
text → IPA, text → respelling, IPA → respelling, each implementing
`data/<code>/phonology.txt` and `data/<code>/respelling/<target>.txt`.
LLM-based steps live under `ancient_language_tts.llm`.

Each engine works on written IPA lines internally. The functions here are
the boundary: `to_ipa` returns a list of IPA tokens (see `ipa_tokens`),
and `respell_ipa` takes one.
"""

from __future__ import annotations

from collections.abc import Sequence

from .. import ipa_tokens, languages
from . import ang, eme, enm, gem_pro, grc, ine_pro, lat_ecc, sla_pro
from .base import Convert, RuleEngine

_ENGINES: dict[str, RuleEngine] = {
    module.ENGINE.source: module.ENGINE
    for module in (lat_ecc, grc, eme, enm, ang, gem_pro, ine_pro, sla_pro)
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


def to_ipa(text: str, source: str) -> list[str]:
    """Rule-based IPA tokens for `text`. Does not depend on a modern language."""
    return ipa_tokens.parse(engine(source).to_ipa(text), source)


def respell(text: str, source: str, target: str) -> str:
    """Rule-based respelling of the source spelling, with no IPA step."""
    return _pick(engine(source), "respell", target)(text)


def respell_ipa(ipa: Sequence[str], source: str, target: str) -> str:
    """Rule-based respelling of an IPA token list."""
    line = ipa_tokens.render(ipa_tokens.ensure(ipa, source))
    return _pick(engine(source), "respell_ipa", target)(line)


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
    "respell",
    "respell_ipa",
    "to_ipa",
]
