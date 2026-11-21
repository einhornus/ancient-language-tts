"""Rule-based Ecclesiastical Latin → Italian against the bundled examples."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from ancient_language_tts import rule_based
from ancient_language_tts.rule_based.lat_ecc_it import direct as direct_mod
from ancient_language_tts.rule_based.lat_ecc_it import ipa as ipa_mod
from ancient_language_tts.rule_based.lat_ecc_it import respelling as respell_mod
import ancient_language_tts as alt

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
        self.assertFalse(rule_based.is_mock("lat-ecc"))

    def test_unknown_target_raises(self) -> None:
        with self.assertRaises(ValueError):
            rule_based.respell("arma", "lat", "en")

    def test_round_trip_short(self) -> None:
        ipa = rule_based.to_ipa("Caesar", "lat-ecc")
        self.assertEqual(ipa, "ˈtʃɛ.sar")
        self.assertEqual(rule_based.respell_ipa(ipa, "lat-ecc", "it").lower(), "cèssar")
        self.assertEqual(rule_based.respell("Caesar", "lat-ecc", "it").lower(), "cèssar")


class ApiRulesTests(unittest.TestCase):
    def test_rules_methods_skip_llm(self) -> None:
        with (
            mock.patch("ancient_language_tts.api.llm_steps.to_ipa") as llm_ipa,
            mock.patch("ancient_language_tts.api.llm_steps.respell") as llm_respell,
        ):
            ipa = alt.to_ipa("Caesar", "lat-ecc", method="rules")
            via_ipa = alt.respell_ipa(ipa, "lat-ecc", "it", method="rules")
            direct = alt.respell("Caesar", "lat-ecc", "it", method="rules")
        llm_ipa.assert_not_called()
        llm_respell.assert_not_called()
        self.assertEqual(ipa, "ˈtʃɛ.sar")
        self.assertEqual(via_ipa.lower(), "cèssar")
        self.assertEqual(direct.lower(), "cèssar")

    def test_bad_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            alt.to_ipa("Caesar", "lat-ecc", method="magic")  # type: ignore[arg-type]

    def test_speak_passes_voice(self) -> None:
        with mock.patch("ancient_language_tts.api.tts.synthesize", return_value=b"audio") as synth:
            audio = alt.speak("Cèssar", "it", "voice123")
        self.assertEqual(audio, b"audio")
        self.assertEqual(synth.call_args.kwargs["voice_id"], "voice123")
        self.assertEqual(synth.call_args.args[:2], ("Cèssar", "it"))


if __name__ == "__main__":
    unittest.main()
