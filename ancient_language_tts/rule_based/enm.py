"""Middle English: the full rule engine (see `enm_rules`)."""

from __future__ import annotations

from . import enm_rules
from .base import RuleEngine

ENGINE = RuleEngine(
    source="enm",
    to_ipa=enm_rules.to_ipa,
    respell={"en": enm_rules.to_english, "nl": enm_rules.to_dutch},
    respell_ipa={"en": enm_rules.ipa_to_english, "nl": enm_rules.ipa_to_dutch},
)
