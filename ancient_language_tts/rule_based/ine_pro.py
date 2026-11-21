"""Proto-Indo-European: the full rule engine (see `ine_pro_rules`)."""

from __future__ import annotations

from . import ine_pro_rules
from .base import RuleEngine

ENGINE = RuleEngine(
    source="ine-pro",
    to_ipa=ine_pro_rules.to_ipa,
    respell={"lt": ine_pro_rules.to_lithuanian},
    respell_ipa={"lt": ine_pro_rules.ipa_to_lithuanian},
)
