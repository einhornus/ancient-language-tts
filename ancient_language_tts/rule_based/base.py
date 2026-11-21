"""What one language's rule engine provides.

Each ancient language has three deterministic conversions:

    to_ipa(text)          source spelling → IPA. Depends on the source only.
    respell[target](text) source spelling → modern spelling, with no IPA step.
    respell_ipa[target](ipa)  IPA → modern spelling.

The last two depend on the modern language, so they are keyed by its code.
`mock` is True for engines that are placeholders: a small grapheme table,
no stress, no syllables. They run and give a rough line, but they are not
the rules in `data/<code>/phonology.txt`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

Convert = Callable[[str], str]


@dataclass(frozen=True)
class RuleEngine:
    source: str
    to_ipa: Convert
    respell: Mapping[str, Convert] = field(default_factory=dict)
    respell_ipa: Mapping[str, Convert] = field(default_factory=dict)
    mock: bool = False

    def targets(self) -> list[str]:
        """Modern languages this engine can respell into (from text or IPA)."""
        return sorted(set(self.respell) | set(self.respell_ipa))
