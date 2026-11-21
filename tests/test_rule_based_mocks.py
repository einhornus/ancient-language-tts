"""Every ancient language has a rule engine, mock or real, for each bundled target."""

from __future__ import annotations

import unittest

from ancient_language_tts import languages, rule_based, rules

SAMPLES = {
    "lat": "Arma virumque canō",
    "lat-ecc": "Pater noster",
    "grc": "ἄνθρωπος φίλος",
    "eme": "knight nation",
    "enm": "knyght",
    "ang": "cyning ċild",
    "gem-pro": "*kuningaz",
    "ine-pro": "*ph₂tḗr",
    "sla-pro": "*gȏrdъ",
}


class EngineCoverageTests(unittest.TestCase):
    def test_every_language_has_an_engine(self) -> None:
        for item in languages.ANCIENT:
            with self.subTest(source=item.code):
                self.assertEqual(rule_based.engine(item.code).source, item.code)

    def test_engine_covers_bundled_targets(self) -> None:
        for item in languages.ANCIENT:
            engine = rule_based.engine(item.code)
            for target in rules.targets(item.code):
                with self.subTest(source=item.code, target=target):
                    self.assertIn(target, engine.respell)
                    self.assertIn(target, engine.respell_ipa)

    def test_three_conversions_run(self) -> None:
        for source, text in SAMPLES.items():
            ipa = rule_based.to_ipa(text, source)
            self.assertTrue(ipa, source)
            for target in rule_based.engine(source).targets():
                with self.subTest(source=source, target=target):
                    self.assertTrue(rule_based.respell_ipa(ipa, source, target))
                    self.assertTrue(rule_based.respell(text, source, target))

    def test_only_lat_ecc_is_real(self) -> None:
        real = [item.code for item in languages.ANCIENT if not rule_based.is_mock(item.code)]
        self.assertEqual(real, ["lat-ecc"])

    def test_mock_samples(self) -> None:
        self.assertEqual(rule_based.to_ipa("φίλος", "grc"), "pʰilos")
        self.assertEqual(rule_based.respell("φίλος", "grc", "el"), "Φιλος")
        self.assertEqual(rule_based.respell_ipa("ˈku.niŋ.gɑz", "gem-pro", "is"), "Kuningaz")


if __name__ == "__main__":
    unittest.main()
