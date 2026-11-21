"""Speak the North Wind and the Sun in Ecclesiastical Latin, with the bundled Italian voice.

    pip install -e .
    # OPENROUTER_API_KEY and ELEVENLABS_API_KEY in the environment, or in .env
    python examples/quickstart.py
"""

import json
import logging
from pathlib import Path

import ancient_language_tts as alt

logging.basicConfig(level=logging.INFO, format="%(message)s")

lines = json.loads((Path(__file__).parent / "lines.json").read_text(encoding="utf-8"))
text = lines["lat-ecc"]

# Rule-based, no LLM calls. Either straight from the spelling...
respelled = alt.respell(text, "lat-ecc", "it", method="rules")
# ...or through IPA:
ipa = alt.to_ipa(text, "lat-ecc", method="rules")
respelled_via_ipa = alt.respell_ipa(ipa, "lat-ecc", "it", method="rules")

# The LLM pipeline is the same calls with method="llm" (and a tier):
#   ipa = alt.to_ipa(text, "lat-ecc", method="llm", tier="low")
#   respelled = alt.respell_ipa(ipa, "lat-ecc", "it", method="llm", text=text, tier="low")

output = Path(__file__).resolve().parent.parent / "out" / "lat.mp3"
audio = alt.speak(respelled, "it", output=output)

print(json.dumps(
    {"ipa": ipa, "respelling": respelled, "via_ipa": respelled_via_ipa, "path": str(output), "bytes": len(audio)},
    indent=2,
    ensure_ascii=False,
))
