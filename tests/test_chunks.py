"""Sentence splits that keep one IPA call inside the length limit."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from ancient_language_tts import chunks, config
from ancient_language_tts import api
from ancient_language_tts.llm.ipa import to_ipa


class SplitTests(unittest.TestCase):
    def test_a_piece_is_not_split_again(self) -> None:
        pieces = chunks.split("aa. bb. cc. dd.", limit=10)
        again = [part for piece in pieces for part in chunks.split(piece, limit=10)]
        self.assertEqual(again, pieces)

    def test_short_passage_stays_whole(self) -> None:
        text = "A. B. C. D."
        self.assertEqual(chunks.split(text, limit=100), [text])

    def test_one_sentence_stays_whole_when_longer_than_the_limit(self) -> None:
        text = "one sentence without a boundary in it"
        self.assertEqual(chunks.split(text, limit=10), [text])

    def test_empty(self) -> None:
        self.assertEqual(chunks.split("   ", limit=10), [])

    def test_period_without_a_space_is_not_a_boundary(self) -> None:
        text = "Hello.World stays. Next."
        self.assertEqual(chunks.split(text, limit=5), ["Hello.World stays.", "Next."])

    def test_newline_is_not_a_boundary(self) -> None:
        text = "Hello.\nWorld is long."
        self.assertEqual(chunks.split(text, limit=5), [text])

    def test_cuts_at_the_boundary_closest_to_the_middle(self) -> None:
        # 15 characters. The middle boundary leaves two pieces of 7, both
        # inside a limit of 10, so the halves are not cut again.
        text = "aa. bb. cc. dd."
        self.assertEqual(chunks.split(text, limit=10), ["aa. bb.", "cc. dd."])

    def test_halves_are_cut_again_while_they_are_still_too_long(self) -> None:
        text = "aa. bb. cc. dd."
        self.assertEqual(chunks.split(text, limit=6), ["aa.", "bb.", "cc.", "dd."])

    def test_question_mark_and_exclamation_mark(self) -> None:
        self.assertEqual(chunks.split("Go? Stop! Now.", limit=8), ["Go?", "Stop!", "Now."])

    def test_tie_uses_the_earlier_boundary(self) -> None:
        # 12 characters, middle at 6. Boundaries at 4 and 8 are equally close.
        # The earlier cut leaves a right half that is still over the limit.
        self.assertEqual(chunks.split("ab. cd. efg.", limit=7), ["ab.", "cd.", "efg."])

    def test_ellipsis_ends_at_the_last_period(self) -> None:
        self.assertEqual(chunks.split("Wait... Next sentence.", limit=5), ["Wait...", "Next sentence."])

    def test_limit_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            chunks.split("A. B.", limit=0)

    def test_setting_is_the_default_limit(self) -> None:
        text = "aa. bb. cc. dd."
        with config.use(config.resolve(max_chars=10)):
            self.assertEqual(chunks.split(text), ["aa. bb.", "cc. dd."])

    def test_environment_sets_the_limit(self) -> None:
        with mock.patch.dict(os.environ, {"ANCIENT_TTS_MAX_CHARS": "42"}):
            self.assertEqual(config.resolve().max_chars, 42)

    def test_bad_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            config.resolve(max_chars=0)


class PipelineTests(unittest.TestCase):
    def test_llm_respell_runs_ipa_and_respelling_on_each_piece(self) -> None:
        text = "aa. bb. cc. dd."

        def fake_ipa(part: str, source: str, *, tier: str = "high", **settings: object) -> str:
            return f"/{part}/"

        def fake_respell(
            part: str,
            ipa: str,
            source: str,
            target: str = "en",
            *,
            tier: str = "high",
            **settings: object,
        ) -> str:
            self.assertEqual(ipa, f"/{part}/")
            return part.upper()

        with (
            mock.patch("ancient_language_tts.api.llm_steps.to_ipa", side_effect=fake_ipa) as ipa,
            mock.patch("ancient_language_tts.api.llm_steps.respell", side_effect=fake_respell) as spelling,
        ):
            line = api.respell(text, "lat", "it", method="llm", max_chars=10)

        self.assertEqual([call.args[0] for call in ipa.call_args_list], ["aa. bb.", "cc. dd."])
        self.assertEqual([call.args[0] for call in spelling.call_args_list], ["aa. bb.", "cc. dd."])
        self.assertEqual(line, "AA. BB. CC. DD.")

    def test_to_ipa_joins_the_pieces(self) -> None:
        with mock.patch("ancient_language_tts.llm.ipa._ensemble", side_effect=lambda text, source, tier: text[::-1]):
            line = to_ipa("aa. bb. cc. dd.", "lat", max_chars=10)
        self.assertEqual(line, ".bb .aa .dd .cc")


if __name__ == "__main__":
    unittest.main()
