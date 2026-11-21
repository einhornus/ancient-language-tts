"""Old English: the full rule engine (see `ang_rules`)."""

from __future__ import annotations

from . import ang_rules
from .base import RuleEngine

ENGINE = RuleEngine(
    source="ang",
    to_ipa=ang_rules.to_ipa,
    respell={"is": ang_rules.to_icelandic, "nl": ang_rules.to_dutch},
    respell_ipa={"is": ang_rules.ipa_to_icelandic, "nl": ang_rules.ipa_to_dutch},
)
