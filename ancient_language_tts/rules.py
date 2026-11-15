"""Phonetic rules, IPA examples, and respelling rules, shipped as text files.

`data/<code>/phonology.txt` is the system-prompt section for the IPA call.
`data/<code>/examples.txt` is the to-IPA examples included in that prompt.
`data/<modern>/phonology.txt` is how that language's spelling is read aloud.
`data/<ancient>/respelling/<modern>.txt` is how that ancient phonology maps
onto that spelling. `data/<ancient>/respelling/<modern>.examples.txt` shows
that mapping as the original text, then the IPA, then the respelling. The
pair file is what makes the second call depend on both languages.
"""

from __future__ import annotations

from pathlib import Path

from . import languages

_ROOT = Path(__file__).resolve().parent
_DATA = _ROOT / "data"
_CACHE: dict[Path, str] = {}


def _read(relative: str) -> str:
    path = _DATA / relative
    if not path.is_file():
        raise ValueError(f"missing rules file data/{relative}")
    cached = _CACHE.get(path)
    if cached is None:
        cached = path.read_text(encoding="utf-8").strip()
        _CACHE[path] = cached
    return cached


def phonology(source: str) -> str:
    """The phonetic rules for one ancient language."""
    language = languages.ancient(source)
    return _read(f"{language.code}/phonology.txt")


def examples(source: str) -> str:
    """Worked to-IPA lines for one ancient language."""
    language = languages.ancient(source)
    return _read(f"{language.code}/examples.txt")


def orthography(target: str) -> str:
    """How one modern language's spelling is read by a voice."""
    language = languages.modern(target)
    return _read(f"{language.code}/phonology.txt")


def targets(source: str) -> list[str]:
    """Modern language codes that have a pair file for `source`."""
    language = languages.ancient(source)
    folder = _DATA / language.code / "respelling"
    found: list[str] = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.txt")):
            if path.name.endswith(".examples.txt"):
                continue
            found.append(path.stem)
    return found


def pair(source: str, target: str) -> str:
    """The respelling rules for one ancient language read by one modern voice."""
    ancient = languages.ancient(source)
    modern = languages.modern(target)
    relative = f"{ancient.code}/respelling/{modern.code}.txt"
    path = _DATA / relative
    if not path.is_file():
        have = ", ".join(targets(ancient.code)) or "(none)"
        raise ValueError(
            f"no respelling rules for {ancient.code} → {modern.code}. "
            f"Bundled targets for {ancient.code}: {have}."
        )
    return _read(relative)


def pair_examples(source: str, target: str) -> str:
    """Worked original, IPA, and respelling lines for one language pair.

    Empty when that pair has rules and no examples file.
    """
    ancient = languages.ancient(source)
    modern = languages.modern(target)
    relative = f"{ancient.code}/respelling/{modern.code}.examples.txt"
    if not (_DATA / relative).is_file():
        return ""
    return _read(relative)
