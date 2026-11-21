"""Proto-Slavic: the full rule engine (see `sla_pro_rules`)."""

from __future__ import annotations

from . import sla_pro_rules
from .base import RuleEngine

ENGINE = RuleEngine(
    source="sla-pro",
    to_ipa=sla_pro_rules.to_ipa,
    respell={"uk": sla_pro_rules.to_ukrainian},
    respell_ipa={"uk": sla_pro_rules.ipa_to_ukrainian},
)
