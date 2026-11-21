"""Language codes.

Ancient languages are the ones this package transcribes. `lat` is Classical
Latin and `lat-ecc` is Ecclesiastical Latin in the Italian school. `grc` is
Koine Greek of the first century AD. Other stages of those languages are different phonologies
and are not folded into these codes. `eme` and `lat-ecc` are project codes:
ISO 639 has no entry for either. The proto-language codes are the ones
Wiktionary uses.

Modern languages are the ones a voice can speak. A modern language needs a
spelling table (`data/<code>/phonology.txt`) and, for each ancient language
it can read, a pair file (`data/<ancient>/respelling/<code>.txt`) and an
examples file (`data/<ancient>/respelling/<code>.examples.txt`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AncientLanguage:
    code: str
    name: str
    era: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "name": self.name, "era": self.era}


@dataclass(frozen=True)
class ModernLanguage:
    code: str
    name: str
    elevenlabs_code: str
    voice_id: str
    voice_name: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "name": self.name,
            "voice_id": self.voice_id,
            "voice_name": self.voice_name,
        }


ANCIENT: tuple[AncientLanguage, ...] = (
    AncientLanguage("lat", "Classical Latin", "c. 100 BC–AD 100"),
    AncientLanguage(
        "lat-ecc",
        "Ecclesiastical Latin",
        "Italian school, Roman pronunciation",
    ),
    AncientLanguage("grc", "Koine Greek", "Roman-era Koine, 1st century AD"),
    AncientLanguage("eme", "Early Modern English", "c. 1600 AD"),
    AncientLanguage("enm", "Middle English", "London, c. 1400 AD"),
    AncientLanguage("ang", "Old English", "Late West Saxon, c. 900 AD"),
    AncientLanguage("gem-pro", "Proto-Germanic", "c. 200 AD"),
    AncientLanguage("ine-pro", "Proto-Indo-European", "c. 3000 BC"),
    AncientLanguage("sla-pro", "Proto-Slavic", "Early Common Slavic, c. 500 AD"),
)

# George is the premade voice ElevenLabs documents as the safe default.
# Giovanni is the premade voice with an Italian accent; language_code "it"
# selects Italian letter-to-sound. Either id may be routed to a newer voice
# on their side. ELEVENLABS_VOICE_ID pins another.
MODERN: tuple[ModernLanguage, ...] = (
    ModernLanguage(
        "en",
        "English",
        "en",
        "JBFqnCBsd6RMkjVDRZzb",
        "George",
    ),
    ModernLanguage(
        "it",
        "Italian",
        "it",
        "zcAOhNBS3c14rBihAFp1",
        "Giovanni",
    ),
    # No premade ElevenLabs voice is Dutch. George is multilingual, and
    # language_code "nl" selects Dutch letter-to-sound. Pin a Dutch voice
    # from the voice library with ELEVENLABS_VOICE_ID for a native accent.
    ModernLanguage(
        "nl",
        "Dutch",
        "nl",
        "JBFqnCBsd6RMkjVDRZzb",
        "George",
    ),
    # No premade ElevenLabs voice is Icelandic either. language_code "is"
    # selects Icelandic letter-to-sound on a model that supports it.
    ModernLanguage(
        "is",
        "Icelandic",
        "is",
        "JBFqnCBsd6RMkjVDRZzb",
        "George",
    ),
    # No premade ElevenLabs voice is Greek. language_code "el" selects
    # Modern Greek letter-to-sound, which reads the Koine respelling.
    ModernLanguage(
        "el",
        "Modern Greek",
        "el",
        "JBFqnCBsd6RMkjVDRZzb",
        "George",
    ),
    # No premade ElevenLabs voice is Lithuanian. language_code "lt" selects
    # Lithuanian letter-to-sound on a model that supports it; it reads the
    # Proto-Indo-European respelling.
    ModernLanguage(
        "lt",
        "Lithuanian",
        "lt",
        "JBFqnCBsd6RMkjVDRZzb",
        "George",
    ),
    # No premade ElevenLabs voice is Ukrainian. language_code "uk" selects
    # Ukrainian letter-to-sound on a model that supports it; it reads the
    # Proto-Slavic respelling.
    ModernLanguage(
        "uk",
        "Ukrainian",
        "uk",
        "JBFqnCBsd6RMkjVDRZzb",
        "George",
    ),
)

_ANCIENT_ALIASES = {
    "la": "lat",
    "latin": "lat",
    "classical-latin": "lat",
    "ecclesiastical-latin": "lat-ecc",
    "ecclesiastical": "lat-ecc",
    "church-latin": "lat-ecc",
    "ancient-greek": "grc",
    "greek": "grc",
    "koine": "grc",
    "koine-greek": "grc",
    "biblical-greek": "grc",
    "new-testament-greek": "grc",
    "early-modern-english": "eme",
    "middle-english": "enm",
    "old-english": "ang",
    "anglo-saxon": "ang",
    "oe": "ang",
    "proto-germanic": "gem-pro",
    "pgmc": "gem-pro",
    "proto-indo-european": "ine-pro",
    "pie": "ine-pro",
    "proto-slavic": "sla-pro",
    "common-slavic": "sla-pro",
    "psl": "sla-pro",
}

_MODERN_ALIASES = {
    "english": "en",
    "italian": "it",
    "dutch": "nl",
    "nederlands": "nl",
    "icelandic": "is",
    "islenska": "is",
    "íslenska": "is",
    "modern-greek": "el",
    "ellinika": "el",
    "ελληνικά": "el",
    "lithuanian": "lt",
    "lietuviu": "lt",
    "lietuvių": "lt",
    "ukrainian": "uk",
    "ukrainska": "uk",
    "українська": "uk",
}

_ANCIENT_BY_CODE = {item.code: item for item in ANCIENT}
_MODERN_BY_CODE = {item.code: item for item in MODERN}


def _normalize(value: str) -> str:
    parts = value.strip().lower().replace("_", "-").replace(" ", "-").split("-")
    return "-".join(part for part in parts if part)


def ancient(code: str) -> AncientLanguage:
    """The ancient language `code` names. Accepts the code or the English name."""
    key = _ANCIENT_ALIASES.get(_normalize(code), _normalize(code))
    try:
        return _ANCIENT_BY_CODE[key]
    except KeyError:
        codes = ", ".join(item.code for item in ANCIENT)
        raise ValueError(f"unknown ancient language {code!r}. Codes: {codes}") from None


def modern(code: str) -> ModernLanguage:
    """The modern language `code` names. Accepts the code or the English name."""
    key = _MODERN_ALIASES.get(_normalize(code), _normalize(code))
    try:
        return _MODERN_BY_CODE[key]
    except KeyError:
        codes = ", ".join(item.code for item in MODERN)
        raise ValueError(f"unknown modern language {code!r}. Codes: {codes}") from None


def ancient_languages() -> list[dict[str, str]]:
    return [item.as_dict() for item in ANCIENT]


def modern_languages() -> list[dict[str, str]]:
    return [item.as_dict() for item in MODERN]
