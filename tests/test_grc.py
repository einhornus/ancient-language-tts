"""Koine Greek → el: IPA, respelling and voice quality, scored by the phoneme recognizer.

    pytest tests/test_grc.py      or      python tests/test_grc.py

The report is printed and written to out/tests/grc.el.json (see harness.py).
"""

from harness import run

SOURCE = "grc"
TARGET = "el"

SENTENCES = 10          # the first N sentences of data/ang/texts.json
VOICES_PER_GENDER = 15    # the first M males + the first M females of voices/nl.json
TIER = "high"             # LLM tier: "high", "medium", "low"
EFFORT = "low"           # LLM effort: "ultra", "high", "medium", "low", "basic"
MAX_MISMATCH = None      # fail when the pooled mismatch is above this; None never fails


def test_grc():
    run(
        SOURCE,
        TARGET,
        sentences=SENTENCES,
        voices_per_gender=VOICES_PER_GENDER,
        tier=TIER,
        effort=EFFORT,
        max_mismatch=MAX_MISMATCH,
    )


if __name__ == "__main__":
    test_grc()
