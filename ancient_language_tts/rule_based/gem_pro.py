"""Proto-Germanic: the full rule engine (see `gem_pro_rules`)."""

from __future__ import annotations

from . import gem_pro_rules
from .base import RuleEngine

ENGINE = RuleEngine(
    source="gem-pro",
    to_ipa=gem_pro_rules.to_ipa,
    respell={"is": gem_pro_rules.to_icelandic, "nl": gem_pro_rules.to_dutch},
    respell_ipa={"is": gem_pro_rules.respell_ipa_is, "nl": gem_pro_rules.respell_ipa_nl},
)
