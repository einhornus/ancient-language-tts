"""Rule-based Ecclesiastical Latin → Italian against the bundled examples."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from ancient_language_tts import rule_based
from ancient_language_tts.rule_based.lat_ecc_it import direct as direct_mod
from ancient_language_tts.rule_based.lat_ecc_it import ipa as ipa_mod
from ancient_language_tts.rule_based.lat_ecc_it import respelling as respell_mod
from ancient_language_tts.speak import speak

_DATA = Path(__file__).resolve().parents[1] / "ancient_language_tts" / "data" / "lat-ecc"


def _ipa_examples() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in (_DATA / "examples.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "→" not in line:
            continue
        left, right = line.split("→", 1)
        rows.append((left.strip(), right.strip()))
    return rows


def _respell_examples() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    path = _DATA / "respelling" / "it.examples.txt"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.count("→") < 2:
            continue
        left, mid, right = [p.strip() for p in line.split("→", 2)]
        rows.append((left, mid, right))
    return rows


class LatEccIpaTests(unittest.TestCase):
    def test_examples(self) -> None:
        for src, want in _ipa_examples():
            with self.subTest(src=src[:40]):
                self.assertEqual(ipa_mod.to_ipa(src), want)


class LatEccRespellingTests(unittest.TestCase):
    def test_examples(self) -> None:
        for _src, ipa, want in _respell_examples():
            with self.subTest(ipa=ipa[:40]):
                self.assertEqual(respell_mod.respell(ipa).lower(), want.lower())


class LatEccDirectTests(unittest.TestCase):
    def test_examples_skip_ipa(self) -> None:
        for src, _ipa, want in _respell_examples():
            with self.subTest(src=src[:40]):
                self.assertEqual(direct_mod.to_italian(src).lower(), want.lower())


class RuleBasedRegistryTests(unittest.TestCase):
    def test_lat_ecc_it_is_available(self) -> None:
        self.assertTrue(rule_based.has("lat-ecc", "it"))
        self.assertIn(("lat-ecc", "it"), rule_based.available())

    def test_unknown_pair_raises(self) -> None:
        with self.assertRaises(ValueError):
            rule_based.convert("arma", "lat", "en")

    def test_convert_round_trip_short(self) -> None:
        ipa, spelling = rule_based.convert("Caesar", "lat-ecc", "it")
        self.assertEqual(ipa, "ˈtʃɛ.sar")
        self.assertEqual(spelling.lower(), "cèssar")


class SpeakRulesTests(unittest.TestCase):
    def test_use_rules_skips_llm(self) -> None:
        with (
            mock.patch("ancient_language_tts.speak.ipa_step.to_ipa") as llm_ipa,
            mock.patch("ancient_language_tts.speak.respell_step.respell") as llm_respell,
            mock.patch("ancient_language_tts.speak.tts.synthesize", return_value=b"audio"),
        ):
            speech = speak("Caesar", source="lat-ecc", target="it", use_rules=True)
        llm_ipa.assert_not_called()
        llm_respell.assert_not_called()
        self.assertEqual(speech.ipa, "ˈtʃɛ.sar")
        self.assertEqual(speech.respelling.lower(), "cèssar")
        self.assertEqual(speech.audio, b"audio")

    def test_use_rules_without_engine_raises(self) -> None:
        with self.assertRaises(ValueError):
            speak("hello", source="ang", target="en", use_rules=True)

    def test_direct_flag_skips_ipa(self) -> None:
        with (
            mock.patch("ancient_language_tts.speak.rule_based.to_ipa") as to_ipa,
            mock.patch("ancient_language_tts.speak.tts.synthesize", return_value=b"audio"),
        ):
            speech = speak("Caesar", source="lat-ecc", target="it", through_ipa=False)
        to_ipa.assert_not_called()
        self.assertEqual(speech.ipa, "")
        self.assertEqual(speech.respelling.lower(), "cèssar")


if __name__ == "__main__":
    unittest.main()
