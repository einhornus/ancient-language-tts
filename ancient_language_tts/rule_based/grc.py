"""Koine Greek (first century AD): the full rule engine (see `grc_rules`)."""

from __future__ import annotations

from . import grc_rules
from .base import RuleEngine

ENGINE = RuleEngine(
    source="grc",
    to_ipa=grc_rules.to_ipa,
    respell={"el": grc_rules.to_modern_greek},
    respell_ipa={"el": grc_rules.ipa_to_modern_greek},
)
