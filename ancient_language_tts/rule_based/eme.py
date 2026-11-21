"""Early Modern English: the full rule engine (see `eme_rules`)."""

from __future__ import annotations

from . import eme_rules
from .base import RuleEngine

ENGINE = RuleEngine(
    source="eme",
    to_ipa=eme_rules.to_ipa,
    respell={"en": eme_rules.to_english},
    respell_ipa={"en": eme_rules.respell},
)
