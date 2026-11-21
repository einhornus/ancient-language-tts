"""Run the North Wind and the Sun through every ancient language and its modern targets.

    pip install -e .
    # OPENROUTER_API_KEY and ELEVENLABS_API_KEY in the environment, or in .env
    python examples/quickstart.py                  # rules, with audio
    python examples/quickstart.py --no-audio       # rules only, no keys needed
    python examples/quickstart.py --method llm --tier low --only grc lat

For each language and each target, this runs the three conversions
(to_ipa, respell_ipa, respell) and speaks the respelling into
out/<source>.<target>.mp3. Every language except lat-ecc uses a mock
rule engine, so its lines are rough.
"""

import argparse
import json
import logging
from pathlib import Path

import ancient_language_tts as alt

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "out"


def targets(source: str, method: str) -> list[str]:
    if method == "rules":
        return alt.rule_based.engine(source).targets()
    return alt.targets(source)


def run(source: str, text: str, method: str, tier: str, audio: bool) -> list[dict]:
    results = []
    ipa = alt.to_ipa(text, source, method=method, tier=tier)
    for target in targets(source, method):
        via_ipa = alt.respell_ipa(ipa, source, target, method=method, text=text, tier=tier)
        direct = alt.respell(text, source, target, method=method, tier=tier)
        result = {
            "source": source,
            "target": target,
            "mock": alt.rule_based.is_mock(source) if method == "rules" else None,
            "ipa": ipa,
            "respell_ipa": via_ipa,
            "respell": direct,
        }
        if audio:
            output = OUT / f"{source}.{target}.mp3"
            result["path"] = str(output)
            result["bytes"] = len(alt.speak(via_ipa, target, output=output))
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=alt.METHODS, default="rules")
    parser.add_argument("--tier", choices=("high", "medium", "low"), default="low")
    parser.add_argument("--no-audio", action="store_true", help="skip ElevenLabs")
    parser.add_argument("--only", nargs="+", metavar="CODE", help="ancient-language codes to run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lines = json.loads((ROOT / "lines.json").read_text(encoding="utf-8"))
    sources = args.only or [item["code"] for item in alt.ancient_languages()]

    report, failed = [], []
    for source in sources:
        if source not in lines:
            failed.append({"source": source, "error": "no passage in lines.json"})
            continue
        try:
            report.extend(run(source, lines[source], args.method, args.tier, not args.no_audio))
        except (alt.TTSError, ValueError) as exc:
            failed.append({"source": source, "error": str(exc)})

    print(json.dumps({"results": report, "failed": failed}, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
