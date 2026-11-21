"""Command line. One subcommand per step of the pipeline.

    ancient-language-tts ipa "Arma virumque cano" --source lat --method llm
    ancient-language-tts respell "Caesar" --source lat-ecc --target it --method rules
    ancient-language-tts respell-ipa "ˈpʰi.los" --source grc --target el --method llm
    ancient-language-tts speak "Cèssar" --target it --out out/caesar.mp3
    ancient-language-tts languages
    ancient-language-tts rules lat
    ancient-language-tts rules lat it

A positional argument that is a path to a file is read as UTF-8. Anything
else is the text itself. The nearest `.env` at or above the current folder
is read, so the two keys can live there.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from . import api, languages, rule_based, rules
from .llm import ipa as llm_ipa
from .llm import respelling as llm_respelling


def _load_text(value: str) -> str:
    path = Path(value)
    looks_like_path = path.suffix.lower() in {".txt", ".md"} or value.startswith(("./", ".\\", "/", "\\"))
    if path.is_file():
        return path.read_text(encoding="utf-8-sig")
    if looks_like_path or (path.suffix.lower() in {".txt", ".md"} and not path.exists()):
        raise ValueError(f"no such file: {path}")
    return value


def _dump(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _llm_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tier": args.tier,
        "openrouter_api_key": args.openrouter_key,
        "models": args.models,
        "cache_dir": args.cache_dir,
        "cache": False if args.no_cache else None,
    }


def _ipa(args: argparse.Namespace) -> int:
    print(api.to_ipa(
        _load_text(args.text),
        args.source,
        args.method,
        max_chars=args.max_chars,
        **_llm_settings(args),
    ))
    return 0


def _respell(args: argparse.Namespace) -> int:
    print(api.respell(
        _load_text(args.text),
        args.source,
        args.target,
        args.method,
        max_chars=args.max_chars,
        **_llm_settings(args),
    ))
    return 0


def _respell_ipa(args: argparse.Namespace) -> int:
    print(api.respell_ipa(
        _load_text(args.ipa),
        args.source,
        args.target,
        args.method,
        text=_load_text(args.text) if args.text else None,
        **_llm_settings(args),
    ))
    return 0


def _speak(args: argparse.Namespace) -> int:
    audio = api.speak(
        _load_text(args.text),
        args.target,
        args.voice,
        output=args.out,
        seed=args.seed,
        elevenlabs_api_key=args.elevenlabs_key,
        elevenlabs_model=args.elevenlabs_model,
        cache_dir=args.cache_dir,
        cache=False if args.no_cache else None,
    )
    _dump({"path": args.out, "bytes": len(audio), "target": args.target})
    return 0


def _languages(_args: argparse.Namespace) -> int:
    print("ancient")
    for item in languages.ANCIENT:
        have = ", ".join(rules.targets(item.code)) or "(none)"
        engine = rule_based.engine(item.code)
        kind = "mock" if engine.mock else "rules"
        ruled = ", ".join(engine.targets()) or "(none)"
        print(
            f"  {item.code:<10} {item.name:<24} {item.era:<44} "
            f"llm: {have:<8} {kind}: {ruled}"
        )
    print("modern")
    for item in languages.MODERN:
        print(f"  {item.code:<10} {item.name:<24} voice: {item.voice_name} ({item.voice_id})")
    return 0


def _rules(args: argparse.Namespace) -> int:
    if args.target:
        print(llm_respelling.system_prompt(args.source, args.target))
    else:
        print(llm_ipa.system_prompt(args.source))
    return 0


def _add_limit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-chars",
        type=int,
        help="source characters one IPA call may see (default 1000)",
    )


def _add_method(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--method",
        choices=api.METHODS,
        default="llm",
        help="llm (default) or rules: the deterministic engine, no LLM calls",
    )


def _add_llm(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--openrouter-key", help="OpenRouter key (default: OPENROUTER_API_KEY)")
    parser.add_argument("--tier", choices=("high", "medium", "low"), default="high", help="LLM cost tier")
    parser.add_argument("--models", help="comma-separated OpenRouter LLM ids, replacing the tier list")
    parser.add_argument("--cache-dir", help="answer and audio cache (default .cache/ancient-language-tts)")
    parser.add_argument("--no-cache", action="store_true", help="ignore and do not write the caches")


def _add_speech(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--elevenlabs-key", help="ElevenLabs key (default: ELEVENLABS_API_KEY)")
    parser.add_argument("--voice", help="ElevenLabs voice id (default: the target language's voice)")
    parser.add_argument("--seed", type=int, help="best-effort repeatable take, 0 to 4294967295")
    parser.add_argument("--cache-dir", help="answer and audio cache (default .cache/ancient-language-tts)")
    parser.add_argument("--no-cache", action="store_true", help="ignore and do not write the caches")


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass
    parser = argparse.ArgumentParser(
        prog="ancient-language-tts",
        description="Text-to-speech for ancient languages, through a modern voice.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v for each step, -vv for every prompt")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ipa", help="ancient text → IPA")
    p.add_argument("text", help="the passage, or a UTF-8 .txt file")
    p.add_argument("--source", required=True, help="ancient-language code, e.g. lat")
    _add_method(p)
    _add_llm(p)
    _add_limit(p)
    p.set_defaults(run=_ipa)

    p = sub.add_parser("respell", help="ancient text → modern spelling")
    p.add_argument("text", help="the passage, or a UTF-8 .txt file")
    p.add_argument("--source", required=True, help="ancient-language code")
    p.add_argument("--target", default="en", help="modern-language code (default en)")
    _add_method(p)
    _add_llm(p)
    _add_limit(p)
    p.set_defaults(run=_respell)

    p = sub.add_parser("respell-ipa", help="IPA → modern spelling")
    p.add_argument("ipa", help="the IPA, or a UTF-8 .txt file")
    p.add_argument("--source", required=True, help="ancient-language code the IPA came from")
    p.add_argument("--target", default="en", help="modern-language code (default en)")
    p.add_argument("--text", help="the original passage, for word boundaries (LLM only)")
    _add_method(p)
    _add_llm(p)
    p.set_defaults(run=_respell_ipa)

    p = sub.add_parser("speak", help="speak text that is already in a modern spelling")
    p.add_argument("text", help="the modern-language spelling, or a UTF-8 .txt file")
    p.add_argument("--target", default="en", help="modern-language code (default en)")
    p.add_argument("--out", required=True, help="audio file to write")
    p.add_argument("--elevenlabs-model", help="ElevenLabs model (default eleven_v4)")
    _add_speech(p)
    p.set_defaults(run=_speak)

    p = sub.add_parser("languages", help="list language codes, eras, and bundled targets")
    p.set_defaults(run=_languages)

    p = sub.add_parser("rules", help="print the system prompt for a language or a pair")
    p.add_argument("source", help="ancient-language code")
    p.add_argument("target", nargs="?", help="modern-language code; omit to print the phonetic rules")
    p.set_defaults(run=_rules)

    args = parser.parse_args(argv)
    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pydantic_ai").setLevel(logging.WARNING)
    try:
        return args.run(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ancient-language-tts: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
