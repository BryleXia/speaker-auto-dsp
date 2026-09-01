"""Target-curve bass trim tests (car_eq.apply_bass_trim / build_audiofrog_target).

2026-09-01 user decision: the official Audiofrog bass shelf (+9.65 dB @ 20 Hz
rel 1 kHz, transition extending to ~316 Hz) is too hot for this listener.
The trim must:

  1. Pull the 20 Hz shelf down by ~5 dB (user-specified comfort level).
  2. Keep the LF rise positive everywhere (no dip below the midrange —
     the user explicitly rejected "digging a hole" in the bass).
  3. Stay monotonic through the transition band.
  4. Leave the midrange/treble shape untouched — the official curve's
     1-10 kHz skeleton is the Harman in-car consensus (Wehmeyer, "A Note
     about Target Curves").

Run:  py -3 -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import car_eq  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Approved shape (2026-09-01, shelf trim -5 dB @ fc=75 Hz): trimmed target
# relative to 1 kHz, verified against the real shelf_db response.
APPROVED_POINTS = {
    20.0: 4.74,
    63.0: 3.66,
    100.0: 1.86,
    160.0: 0.86,
    200.0: 0.70,
    1000.0: 0.00,
}


def official_rel_1k(freqs: np.ndarray) -> np.ndarray:
    """Official Audiofrog curve normalised to 0 dB at 1 kHz."""
    target_freq, target_db = car_eq.load_audiofrog_target_csv()
    target_1k = float(np.interp(1000.0, target_freq, target_db))
    return np.interp(freqs, target_freq, target_db) - target_1k


class ApplyBassTrimTest(unittest.TestCase):
    def setUp(self):
        self.freq = np.logspace(np.log10(15), np.log10(20000), 1200)
        self.official = official_rel_1k(self.freq)
        self.trimmed = car_eq.apply_bass_trim(self.freq, self.official)

    def test_pulls_20hz_shelf_down_about_5db(self):
        drop_20 = float(np.interp(20.0, self.freq, self.official - self.trimmed))
        self.assertAlmostEqual(drop_20, 5.0, delta=0.2)

    def test_low_frequency_stays_above_midrange(self):
        # No hole in the bass: the trimmed target must not dip toward/below
        # the midrange anywhere in the transition band.
        band = (self.freq >= 20) & (self.freq <= 400)
        self.assertGreaterEqual(float(self.trimmed[band].min()), 0.4)

    def test_transition_stays_monotonic(self):
        band = (self.freq >= 20) & (self.freq <= 315)
        rises = np.diff(self.trimmed[band])
        self.assertLessEqual(float(rises.max()), 0.05)

    def test_midrange_and_treble_untouched(self):
        band = self.freq >= 500
        self.assertLessEqual(float(np.max(np.abs(self.trimmed[band] - self.official[band]))), 0.05)

    def test_offset_window_mean_hardly_moves(self):
        # The 200-8000 Hz window is re-centred inside make_shared_eq; the
        # trim must not shift its mean by anything audible.
        band = (self.freq >= 200) & (self.freq <= 8000)
        shift = float(np.mean(self.trimmed[band] - self.official[band]))
        self.assertLessEqual(abs(shift), 0.15)

    def test_zero_trim_disables(self):
        untouched = car_eq.apply_bass_trim(self.freq, self.official, trim_db=0)
        np.testing.assert_array_equal(untouched, self.official)


class BuildAudiofrogTargetTest(unittest.TestCase):
    def test_matches_approved_shape(self):
        freq = np.array(sorted(APPROVED_POINTS))
        target = car_eq.build_audiofrog_target(freq, ref_level=0.0)
        for f, expected in APPROVED_POINTS.items():
            self.assertAlmostEqual(float(target[freq == f][0]), expected, delta=0.06,
                                   msg=f"{f:g} Hz off the approved shape")

    def test_official_csv_is_the_source(self):
        # The fallback approximation would drift from these values; this pins
        # the official CSV (shipped at the project root) as the curve source.
        self.assertTrue(car_eq.TARGET_CSV.exists(), f"missing {car_eq.TARGET_CSV}")


class ParseArgsTest(unittest.TestCase):
    def test_bass_trim_defaults(self):
        args = car_eq.parse_args(["--car", "MyCar"])
        self.assertEqual(args.bass_trim, 5.0)
        self.assertEqual(args.bass_trim_fc, 75.0)

    def test_bass_trim_overrides(self):
        args = car_eq.parse_args(["--car", "MyCar", "--bass-trim", "0"])
        self.assertEqual(args.bass_trim, 0.0)
        args = car_eq.parse_args(["--car", "MyCar", "--bass-trim", "4", "--bass-trim-fc", "65"])
        self.assertEqual(args.bass_trim, 4.0)
        self.assertEqual(args.bass_trim_fc, 65.0)


if __name__ == "__main__":
    unittest.main()
