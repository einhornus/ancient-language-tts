"""Classical Latin: the full rule engine (see `lat_rules`)."""

from __future__ import annotations

from . import lat_rules
from .base import RuleEngine

ENGINE = RuleEngine(
    source="lat",
    to_ipa=lat_rules.to_ipa,
    respell=dict(lat_rules.RESPELL),
    respell_ipa=dict(lat_rules.RESPELL_IPA),
)
