"""Early Modern English → en: IPA, respelling and voice quality, scored by the phoneme recognizer.

    pytest tests/test_eme.py      or      python tests/test_eme.py

The report is printed and written to out/tests/eme.en.json (see harness.py).
"""

from harness import run

SOURCE = "eme"
TARGET = "en"

SENTENCES = 10          # the first N sentences of data/eme/texts.json
VOICES_PER_GENDER = 15    # the first M males + the first M females of voices/en.json
TIER = "high"             # LLM tier: "high", "medium", "low"
EFFORT = "low"           # LLM effort: "ultra", "high", "medium", "low", "basic"
MAX_MISMATCH = None      # fail when the pooled mismatch is above this; None never fails


def test_eme():
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
    test_eme()
