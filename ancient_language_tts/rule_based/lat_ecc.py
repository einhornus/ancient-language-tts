"""Ecclesiastical Latin: the full rule engine (see `lat_ecc_it`)."""

from __future__ import annotations

from . import lat_ecc_it
from .base import RuleEngine

ENGINE = RuleEngine(
    source="lat-ecc",
    to_ipa=lat_ecc_it.to_ipa,
    respell={"it": lat_ecc_it.to_italian},
    respell_ipa={"it": lat_ecc_it.respell},
)
