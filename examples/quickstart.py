"""Speak the North Wind and the Sun in Ecclesiastical Latin, with the bundled Italian voice.

    pip install -e .
    # OPENROUTER_API_KEY and ELEVENLABS_API_KEY in the environment, or in .env
    python examples/quickstart.py
"""

import json
import logging
from pathlib import Path

import ancient_language_tts

logging.basicConfig(level=logging.INFO, format="%(message)s")

lines = json.loads((Path(__file__).parent / "lines.json").read_text(encoding="utf-8"))
speech = ancient_language_tts.speak(
    lines["lat-ecc"],
    source="lat-ecc",
    target="it",
    output=Path(__file__).resolve().parent.parent / "out" / "lat.mp3",
    tier="low",
    use_rules = True,
    through_ipa=False
)
print(json.dumps(speech.summary(), indent=2, ensure_ascii=False))
