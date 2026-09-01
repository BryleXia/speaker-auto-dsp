"""Input-format compatibility tests for the REW loading layer.

The pipeline must accept, without any REW settings fiddling:

  1. The authoritative default REW export ("权威格式"): unsmoothed,
     full-resolution linear FFT-bin grid starting at the first bin
     (~0.37 Hz) up to the measurement's end frequency, 3 columns
     (Freq/SPL/Phase), possibly GBK-mojibake header comments.
  2. Legacy 96-ppo log exports (the May batch) — bit-for-bit unchanged.

Everything is canonicalised in ``parse_rew``: sub-audio bins dropped,
log grids (>= 48 ppo) pass through untouched, anything else is resampled
onto a 96-ppo log grid anchored at the first bin >= 20 Hz so that
default-format and legacy-format files interoperate inside one batch.

Run:  py -3 -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eq_common import estimate_ppo, load_measurements, parse_rew  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AUTH_L = ROOT / "0831导出" / "权威格式" / "L 格式测试 Aug 31.txt"
MAY_L = ROOT / "原始数据" / "L 90-1 May 29.txt"

LINEAR_STEP = 0.3662109375  # 96000 / 262144, REW 256k FFT at 96 kHz
BIN55 = 55 * LINEAR_STEP    # first bin >= 20 Hz in a default export


def make_log_grid(ppo: int, f_max: float = 20000.0, anchor: float = BIN55) -> np.ndarray:
    # REW's ppo exports include the first grid point at or beyond f_max
    # (the May file has 957 points ending at 20,037.85 Hz for a 20 kHz
    # measurement), hence ceil, not floor.
    n = int(np.ceil(np.log2(f_max / anchor) * ppo)) + 1
    return np.array([round(anchor * 2.0 ** (k / ppo), 6) for k in range(n)])


def make_linear_grid(n_bins: int = 54613, step: float = LINEAR_STEP) -> np.ndarray:
    return np.arange(1, n_bins + 1) * step


def make_coarse_octaves_grid(pts_per_oct: int = 8, f_max: float = 20480.0) -> np.ndarray:
    """Constant-Hz spacing inside each octave: dense enough to pass the
    per-octave guard, but with a median step (60 Hz here) so much larger
    than the first point (20 Hz) that the anchor quantises to zero."""
    freqs: list[float] = []
    j = 0
    while 20.0 * 2.0 ** j < f_max:
        lo, hi = 20.0 * 2.0 ** j, min(20.0 * 2.0 ** (j + 1), f_max)
        freqs.extend(np.linspace(lo, hi, pts_per_oct, endpoint=False).tolist())
        j += 1
    freqs.append(f_max)
    return np.array(freqs)


def write_rew_file(path: Path, freqs, spls, *, c_weight: str = "Off",
                   gbk_source: bool = False) -> Path:
    header = [
        "* Measurement data measured by REW V5.40 beta 134",
        "* Format: 256k Log Swept Sine, 1 sweep at -12.0 dBFS",
        "* Dated: 2026 Aug 31 12:00:00",
        "* REW Settings:",
        f"*  C-weighting compensation: {c_weight}",
        "* Measurement: synthetic",
        "* Smoothing: None",
        "* Freq(Hz)\tSPL(dB)\tPhase(degrees)",
    ]
    body = [f"{f:.6f}\t{s:.3f}\t0.0" for f, s in zip(freqs, spls)]
    text = "\n".join(header + body) + "\n"
    if gbk_source:
        # Header bytes REW writes on a Chinese locale, read as garbage by
        # any non-GBK decoder — the parser must survive this.
        raw = ("* Source: 麦克风 (mic), 音量 0.250\n").encode("gbk")
        path.write_bytes(raw + text.encode("ascii"))
    else:
        path.write_text(text, encoding="utf-8")
    return path


def spl_curve(freqs: np.ndarray) -> np.ndarray:
    """Deterministic, smooth, log-linear-ish test signal."""
    return 75.0 + 6.0 * np.log2(freqs / 20.0) + 2.0 * np.sin(np.log2(freqs) * 3.0)


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)


class TestDefaultLinearExport(TempCase):
    def test_resamples_to_96ppo_grid_anchored_at_first_bin_above_20hz(self) -> None:
        freqs = make_linear_grid()
        path = write_rew_file(self.dir / "lin.txt", freqs, spl_curve(freqs))
        freq, _ = parse_rew(path)
        self.assertGreaterEqual(freq[0], 20.0)
        self.assertAlmostEqual(freq[0], BIN55, places=6)
        self.assertAlmostEqual(estimate_ppo(freq), 96.0, places=1)

    def test_drops_sub_audio_bins(self) -> None:
        freqs = make_linear_grid()
        path = write_rew_file(self.dir / "lin.txt", freqs, spl_curve(freqs))
        freq, _ = parse_rew(path)
        self.assertTrue(np.all(freq >= 20.0))

    def test_resample_preserves_log_linear_curve(self) -> None:
        freqs = make_linear_grid()
        spls = 75.0 + 6.0 * np.log2(freqs / 20.0)  # exactly linear in log-f
        path = write_rew_file(self.dir / "lin.txt", freqs, spls)
        freq, spl = parse_rew(path)
        # The ceil rule puts the last grid point just beyond the source span
        # (edge-held by interp), so assert only inside the source data.
        inside = freq <= freqs[-1]
        self.assertGreater(inside.sum(), 900)
        expected = 75.0 + 6.0 * np.log2(freq[inside] / 20.0)
        # The file stores SPL rounded to 3 decimals (as REW does), so the
        # resampled curve can only be as accurate as that rounding.
        np.testing.assert_allclose(spl[inside], expected, atol=5e-4)

    def test_gbk_mojibake_header_is_tolerated(self) -> None:
        freqs = make_linear_grid()
        path = write_rew_file(self.dir / "gbk.txt", freqs, spl_curve(freqs),
                              gbk_source=True)
        freq, spl = parse_rew(path)
        self.assertEqual(len(freq), len(spl))
        self.assertGreater(len(freq), 900)

    @unittest.skipUnless(AUTH_L.exists(), "authoritative sample not present")
    def test_authoritative_sample_matches_may_grid(self) -> None:
        freq, _ = parse_rew(AUTH_L)
        may_freq, _ = parse_rew(MAY_L)
        n = min(len(freq), len(may_freq))
        self.assertGreaterEqual(n, 900)
        worst = np.max(np.abs(freq[:n] - may_freq[:n]))
        self.assertLessEqual(worst, 1e-3,
                             f"default-export grid drifted {worst:.2e} Hz from legacy grid")


class TestLegacyLogExport(TempCase):
    @unittest.skipUnless(MAY_L.exists(), "May sample not present")
    def test_may_file_passes_through_bit_identical(self) -> None:
        freq, spl = parse_rew(MAY_L)
        raw_freq, raw_spl = [], []
        for line in MAY_L.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                f, s = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            raw_freq.append(f)
            raw_spl.append(s)
        self.assertEqual(freq.tolist(), raw_freq)
        self.assertEqual(spl.tolist(), raw_spl)

    def test_synthetic_96ppo_passes_through(self) -> None:
        freqs = make_log_grid(96)
        path = write_rew_file(self.dir / "log96.txt", freqs, spl_curve(freqs))
        freq, spl = parse_rew(path)
        self.assertEqual(freq.tolist(), freqs.tolist())
        # Expected SPLs must go through the same ":.3f" formatting the file
        # writer used, so the comparison is an exact round-trip.
        expected = [float(f"{s:.3f}") for s in spl_curve(freqs)]
        self.assertEqual(spl.tolist(), expected)

    def test_48ppo_is_accepted(self) -> None:
        freqs = make_log_grid(48)
        path = write_rew_file(self.dir / "log48.txt", freqs, spl_curve(freqs))
        freq, _ = parse_rew(path)
        self.assertEqual(freq.tolist(), freqs.tolist())

    def test_coarse_24ppo_is_rejected(self) -> None:
        freqs = make_log_grid(24)
        path = write_rew_file(self.dir / "log24.txt", freqs, spl_curve(freqs))
        with self.assertRaises(ValueError) as ctx:
            parse_rew(path)
        self.assertIn("ppo", str(ctx.exception))


class TestRejections(TempCase):
    def test_sparse_linear_grid_is_rejected(self) -> None:
        freqs = np.arange(10.0, 20001.0, 10.0)  # ~3 points/oct at 20 Hz
        path = write_rew_file(self.dir / "sparse.txt", freqs, spl_curve(freqs))
        with self.assertRaises(ValueError):
            parse_rew(path)

    def test_sparse_high_frequency_tail_is_rejected(self) -> None:
        # Dense below 10.24 kHz, 3-kHz steps above: the density guard must
        # cover the (partial) top octave too, not stop at the last full one.
        dense = np.arange(20.0, 10240.0, LINEAR_STEP)
        tail = np.arange(13240.0, 20001.0, 3000.0)
        freqs = np.concatenate([dense, tail])
        path = write_rew_file(self.dir / "sparsehf.txt", freqs, spl_curve(freqs))
        with self.assertRaises(ValueError):
            parse_rew(path)

    def test_non_monotonic_frequencies_are_rejected(self) -> None:
        # Majority-duplicate rows drive the median log-step to zero.
        grid = make_log_grid(96)
        freqs = np.concatenate([grid, np.full(1000, grid[-1])])
        path = write_rew_file(self.dir / "dup.txt", freqs, np.full(len(freqs), 75.0))
        with self.assertRaises(ValueError):
            parse_rew(path)

    def test_anchor_quantising_to_zero_raises_cleanly(self) -> None:
        # Coarse per-octave grid: passes the density guard, but the median
        # step (60 Hz) dwarfs freq[0] (20 Hz) so round(20/60)*60 == 0 —
        # this must be a ValueError, not an OverflowError downstream.
        freqs = make_coarse_octaves_grid()
        path = write_rew_file(self.dir / "anchor0.txt", freqs, spl_curve(freqs))
        with self.assertRaises(ValueError):
            parse_rew(path)

    def test_short_span_is_rejected(self) -> None:
        freqs = make_log_grid(96, f_max=3000.0)
        path = write_rew_file(self.dir / "short.txt", freqs, spl_curve(freqs))
        with self.assertRaises(ValueError):
            parse_rew(path)


class TestHeaderWarnings(TempCase):
    def test_c_weighting_on_warns(self) -> None:
        freqs = make_linear_grid()
        path = write_rew_file(self.dir / "cwon.txt", freqs, spl_curve(freqs),
                              c_weight="On")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parse_rew(path)
        self.assertTrue(any("C-weighting" in str(w.message) for w in caught))

    def test_c_weighting_off_is_silent(self) -> None:
        freqs = make_linear_grid()
        path = write_rew_file(self.dir / "cwoff.txt", freqs, spl_curve(freqs),
                              c_weight="Off")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            parse_rew(path)
        self.assertFalse(any("C-weighting" in str(w.message) for w in caught))


class TestBatchInterop(TempCase):
    def test_default_and_legacy_files_mix_in_one_batch(self) -> None:
        legacy = write_rew_file(self.dir / "legacy.txt",
                                make_log_grid(96), spl_curve(make_log_grid(96)))
        lin_freqs = make_linear_grid()
        default = write_rew_file(self.dir / "default.txt", lin_freqs,
                                 spl_curve(lin_freqs))
        freq, spls = load_measurements([legacy, default])
        self.assertEqual(len(spls), 2)
        self.assertEqual(len(freq), len(make_log_grid(96)))

    def test_genuinely_different_grids_still_rejected(self) -> None:
        f96 = write_rew_file(self.dir / "g96.txt", make_log_grid(96),
                             spl_curve(make_log_grid(96)))
        f48 = write_rew_file(self.dir / "g48.txt", make_log_grid(48),
                             spl_curve(make_log_grid(48)))
        with self.assertRaises(ValueError) as ctx:
            load_measurements([f96, f48])
        self.assertIn("grid", str(ctx.exception).lower())

    @unittest.skipUnless(AUTH_L.exists() and MAY_L.exists(), "samples not present")
    def test_real_default_and_may_files_mix_in_one_batch(self) -> None:
        # The synthetic test cannot exercise the relaxed tolerance: its
        # anchor is an exact dyadic float.  The real pair's measured drift
        # is 5.9e-7 Hz — the old 1e-6 tolerance passed it with only 1.7x
        # margin, which any change in file rounding could break.
        freq, spls = load_measurements([MAY_L, AUTH_L])
        self.assertEqual(len(spls), 2)
        self.assertEqual(len(freq), 957)

    def test_same_length_different_anchor_is_rejected(self) -> None:
        # Both 957 points, but anchors 0.09 Hz apart — a tolerance looser
        # than 1e-3 would silently average two different grids.
        g1 = write_rew_file(self.dir / "a.txt", make_log_grid(96, anchor=BIN55),
                            spl_curve(make_log_grid(96)))
        g2 = write_rew_file(self.dir / "b.txt", make_log_grid(96, anchor=20.05),
                            spl_curve(make_log_grid(96, anchor=20.05)))
        with self.assertRaises(ValueError):
            load_measurements([g1, g2])


if __name__ == "__main__":
    unittest.main()
