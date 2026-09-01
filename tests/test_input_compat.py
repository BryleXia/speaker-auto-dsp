"""Input-contract tests for the REW loading layer.

The pipeline accepts exactly one input format — REW's default text export:
raw (FFT-bin) resolution, no smoothing.  ``parse_rew`` canonicalises it onto
a 96-ppo log grid anchored at the first bin >= 20 Hz and rejects everything
else with a clear error:

  * fractional-octave (ppo) exports — already downsampled by REW, and the
    resample is this pipeline's job, done once and deterministically;
  * grids too coarse or too sparse to support 1/6-octave smoothing;
  * truncated exports and non-monotonic frequencies.

Every test runs on synthetic files written into a temp directory — no local
measurement data is referenced, so the suite is meaningful on a fresh clone.

Run:  py -3 -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eq_common import estimate_ppo, load_measurements, parse_rew  # noqa: E402

LINEAR_STEP = 0.3662109375  # 96000 / 262144, REW 256k FFT at 96 kHz
BIN55 = 55 * LINEAR_STEP    # first bin >= 20 Hz in a default export


def make_log_grid(ppo: int, f_max: float = 20000.0, anchor: float = BIN55) -> np.ndarray:
    # REW's ppo exports include the first grid point at or beyond f_max
    # (hence ceil, not floor).
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


def write_rew_file(path: Path, freqs, spls, *, gbk_source: bool = False) -> Path:
    header = [
        "* Measurement data measured by REW V5.40 beta 134",
        "* Format: 256k Log Swept Sine, 1 sweep at -12.0 dBFS",
        "* Dated: 2026 Aug 31 12:00:00",
        "* REW Settings:",
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


class TestFractionalOctaveExportsRejected(TempCase):
    def test_96ppo_export_is_rejected(self) -> None:
        freqs = make_log_grid(96)
        path = write_rew_file(self.dir / "log96.txt", freqs, spl_curve(freqs))
        with self.assertRaises(ValueError) as ctx:
            parse_rew(path)
        self.assertIn("fractional-octave", str(ctx.exception))

    def test_48ppo_export_is_rejected(self) -> None:
        freqs = make_log_grid(48)
        path = write_rew_file(self.dir / "log48.txt", freqs, spl_curve(freqs))
        with self.assertRaises(ValueError) as ctx:
            parse_rew(path)
        self.assertIn("48 ppo", str(ctx.exception))

    def test_error_message_points_at_the_required_format(self) -> None:
        freqs = make_log_grid(96)
        path = write_rew_file(self.dir / "log96.txt", freqs, spl_curve(freqs))
        with self.assertRaises(ValueError) as ctx:
            parse_rew(path)
        self.assertIn("raw resolution", str(ctx.exception))
        self.assertIn("no smoothing", str(ctx.exception))


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


class TestBatchInterop(TempCase):
    def test_two_default_files_mix_in_one_batch(self) -> None:
        freqs = make_linear_grid()
        a = write_rew_file(self.dir / "a.txt", freqs, spl_curve(freqs))
        b = write_rew_file(self.dir / "b.txt", freqs, spl_curve(freqs) + 1.0)
        freq, spls = load_measurements([a, b])
        self.assertEqual(len(spls), 2)
        self.assertAlmostEqual(estimate_ppo(freq), 96.0, places=1)

    def test_different_fft_sizes_sharing_an_anchor_mix(self) -> None:
        # Double bin density (half the step): the first bin >= 20 Hz is then
        # bin 110 = 55 * (step/2), so the anchor reconstruction lands on the
        # same absolute frequency and the canonical grid is identical.
        fine = make_linear_grid(n_bins=2 * 54613, step=LINEAR_STEP / 2)
        normal = make_linear_grid()
        a = write_rew_file(self.dir / "fine.txt", fine, spl_curve(fine))
        b = write_rew_file(self.dir / "normal.txt", normal, spl_curve(normal))
        freq, spls = load_measurements([a, b])
        self.assertEqual(len(spls), 2)
        self.assertAlmostEqual(freq[0], BIN55, places=6)

    def test_different_anchors_are_rejected(self) -> None:
        # A >half-step shift pushes the anchor to the next bin over: same
        # length, different grid — a loose tolerance would silently average
        # two different grids.
        shifted = make_linear_grid() + 0.19
        a = write_rew_file(self.dir / "a.txt", make_linear_grid(),
                           spl_curve(make_linear_grid()))
        b = write_rew_file(self.dir / "b.txt", shifted, spl_curve(shifted))
        with self.assertRaises(ValueError) as ctx:
            load_measurements([a, b])
        self.assertIn("grid", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
