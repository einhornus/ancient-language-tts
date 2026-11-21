"""Proto-Indo-European → lt: IPA, respelling and voice quality, scored by the phoneme recognizer.

    pytest tests/test_ine_pro.py      or      python tests/test_ine_pro.py

The report is printed and written to out/tests/ine-pro.lt.json (see harness.py).
"""

from harness import run

SOURCE = "ine-pro"
TARGET = "lt"

SENTENCES = 100          # the first N sentences of data/ine-pro/texts.json
VOICES_PER_GENDER = 2    # the first M males + the first M females of voices/lt.json
TIER = "low"             # LLM tier: "high", "medium", "low"
EFFORT = "low"           # LLM effort: "ultra", "high", "medium", "low", "basic"
MAX_MISMATCH = None      # fail when the pooled mismatch is above this; None never fails


def test_ine_pro():
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
    test_ine_pro()
