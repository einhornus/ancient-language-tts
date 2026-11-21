"""Classical Latin → it: IPA, respelling and voice quality, scored by the phoneme recognizer.

    pytest tests/test_lat.py      or      python tests/test_lat.py

The report is printed and written to out/tests/lat.it.json (see harness.py).
"""

from harness import run

SOURCE = "lat"
TARGET = "it"

SENTENCES = 10          # the first N sentences of data/lat/texts.json
VOICES_PER_GENDER = 5    # the first M males + the first M females of voices/it.json
TIER = "high"             # LLM tier: "high", "medium", "low"
EFFORT = "low"           # LLM effort: "ultra", "high", "medium", "low", "basic"
MAX_MISMATCH = None      # fail when the pooled mismatch is above this; None never fails


def test_lat():
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
    test_lat()
