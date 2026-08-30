"""
car_eq.py - Shared EQ generator for in-car measurement sets.

This script treats the six measurements in a car folder as one shared 90°
cabin response, energy-averages them, and computes a single GraphicEQ file
using the official Audiofrog target curve (audiofrog_target_curve.csv at the
project root; a built-in approximation is used when the file is absent).

Correction pipeline: variable smoothing (1/6 oct below 2 kHz widening to
1/4 oct above 8 kHz — after REW's variable-smoothing recommendation, gentler
than REW's 1/3 endpoint because the input is already a 6-position spatial
average), then target-minus-measured with the mid-band level offset removed,
bass boosts tapered below 65 Hz, boost limited by the inter-position
consistency ceiling and banned outright at/above 15 kHz (user directive:
hearing protection; that roll-off belongs to the speakers, not the room),
while cuts only respect the -18 dB floor.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eq_common import (
    consistency_boost_ceiling,
    energy_average,
    load_measurements,
    log_interp,
    position_spread,
    require_files,
    save_fig as save_common_fig,
    setup_freq_axis,
    shelf_db,
    smooth_oct,
    write_graphic_eq as write_graphic_eq_file,
)


BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"
TARGET_CSV = BASE / "audiofrog_target_curve.csv"

# Match the reference GraphicEQ frequency grid (graphic_eq_grid_ref.txt, local only).
GRAPHICEQ_FREQS = np.array(
    [
        20,
        21,
        22,
        23,
        24,
        26,
        27,
        29,
        30,
        32,
        34,
        36,
        38,
        40,
        43,
        45,
        48,
        50,
        53,
        56,
        59,
        63,
        66,
        70,
        74,
        78,
        83,
        87,
        92,
        97,
        103,
        109,
        115,
        121,
        128,
        136,
        143,
        151,
        160,
        169,
        178,
        188,
        199,
        210,
        222,
        235,
        248,
        262,
        277,
        292,
        309,
        326,
        345,
        364,
        385,
        406,
        429,
        453,
        479,
        506,
        534,
        565,
        596,
        630,
        665,
        703,
        743,
        784,
        829,
        875,
        924,
        977,
        1032,
        1090,
        1151,
        1216,
        1284,
        1357,
        1433,
        1514,
        1599,
        1689,
        1784,
        1885,
        1991,
        2103,
        2221,
        2347,
        2479,
        2618,
        2766,
        2921,
        3086,
        3260,
        3443,
        3637,
        3842,
        4058,
        4287,
        4528,
        4783,
        5052,
        5337,
        5637,
        5955,
        6290,
        6644,
        7018,
        7414,
        7831,
        8272,
        8738,
        9230,
        9749,
        10298,
        10878,
        11490,
        12137,
        12821,
        13543,
        14305,
        15110,
        15961,
        16860,
        17809,
        18812,
        19871,
    ],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a shared car EQ from L+R measurements.")
    parser.add_argument("--car", required=True, help="Car folder name under the project root (e.g. MyCar).")
    parser.add_argument("--slug", default=None, help="Output filename prefix (default: same as --car).")
    parser.add_argument("--n", type=int, default=6, help="Max number of measurement files to use (default: 6).")
    return parser.parse_args()


def build_measurement_files(data_dir: Path, car_name: str, n: int = 6) -> list[Path]:
    matches = sorted(data_dir.glob(f"L+R {car_name}-*.txt"))
    if not matches:
        raise FileNotFoundError(
            f"No measurement files matching 'L+R {car_name}-*.txt' found in {data_dir}\n"
            f"Expected files like: L+R {car_name}-1.txt, L+R {car_name}-2.txt, ..."
        )
    return matches[:n]


def _parse_target_csv_text(text: str) -> tuple[np.ndarray, np.ndarray]:
    freqs: list[float] = []
    gains: list[float] = []
    reader = csv.reader(line for line in text.splitlines() if line.strip())
    for row in reader:
        if len(row) < 2:
            continue
        try:
            freqs.append(float(row[0]))
            gains.append(float(row[1]))
        except ValueError:
            continue
    if not freqs:
        raise ValueError("Audiofrog target CSV did not contain any numeric data.")
    return np.asarray(freqs), np.asarray(gains)


def _read_target_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return _parse_target_csv_text(path.read_text(encoding="utf-8"))


def _build_audiofrog_fallback() -> tuple[np.ndarray, np.ndarray]:
    """
    Build an Audiofrog-style target curve without external data.

    The curve is anchored to 0 dB at 1 kHz, uses a broad low shelf, and
    applies a mild high-frequency roll-off for a car-friendly target.
    """
    freq = np.logspace(np.log10(20), np.log10(20000), 220)
    low_shelf = shelf_db(freq, fc=105, gain_db=9.0, q=0.65)
    hf_rolloff = np.where(freq > 1000, -1.0 * np.log2(freq / 1000), 0.0)
    target = low_shelf + hf_rolloff
    target -= float(np.interp(1000.0, freq, target))
    return freq, target


def load_audiofrog_target_csv() -> tuple[np.ndarray, np.ndarray]:
    candidates = [TARGET_CSV, OUTPUT_DIR / "cache" / TARGET_CSV.name]
    for path in candidates:
        if path.exists():
            return _read_target_csv(path)
    return _build_audiofrog_fallback()


def build_audiofrog_target(freq: np.ndarray, ref_level: float) -> np.ndarray:
    target_freq, target_db = load_audiofrog_target_csv()
    target_1k = float(np.interp(1000.0, target_freq, target_db))
    return ref_level + np.interp(freq, target_freq, target_db - target_1k)


def variable_smooth(freq: np.ndarray, spl: np.ndarray) -> np.ndarray:
    """Frequency-dependent smoothing: 1/6 octave up to 2 kHz, widening to
    1/4 octave by 8 kHz and staying there above.

    Follows the reasoning behind REW's "variable smoothing" (recommended for
    responses that are to be equalised): inter-position comb filtering grows
    with frequency, so the correction must not chase ever-narrower HF detail.
    The endpoint is 1/4 rather than REW's 1/3 because these measurements are
    already a 6-position energy average — the spatial average has removed most
    of the position-sensitive ripple, and the measured inter-position spread
    above 9 kHz is only 1-2 dB, so the remaining dips are real system response
    that 1/3-octave smoothing would start to erase.
    """
    sm_lo = smooth_oct(freq, spl, 1 / 6)
    sm_hi = smooth_oct(freq, spl, 1 / 4)
    blend = np.clip(np.log2(freq / 2000.0) / np.log2(8000.0 / 2000.0), 0.0, 1.0)
    return sm_lo * (1.0 - blend) + sm_hi * blend


def make_shared_eq(
    freq: np.ndarray,
    raw_spl: np.ndarray,
    target: np.ndarray,
    boost_ceiling: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a single shared GraphicEQ curve.

    Strategy:
      - Variable smoothing (1/6 oct below 2 kHz widening to 1/4 oct above 8 kHz)
        on the energy-averaged cabin response.
      - Remove broadband level offset in 200-8 kHz so the EQ focuses on shape.
      - Allow bass cuts freely, but taper only positive boosts below 65 Hz.
      - Boost is limited by the inter-position consistency ceiling; cuts are
        never limited beyond the -18 dB floor.
      - Never boost at/above 15 kHz (user directive: hearing protection, and
        the roll-off up there is most likely the speakers' own physical limit,
        not something to correct).  Taper the correction to zero from 15 kHz
        to 20 kHz, then resample onto the exact GraphicEQ frequency grid used
        by the reference grid file.
    """
    smoothed = variable_smooth(freq, raw_spl)
    correction = target - smoothed

    mid_mask = (freq >= 200) & (freq <= 8000)
    correction -= float(np.mean(correction[mid_mask]))

    f_lo, f_hi = 20.0, 65.0
    bass_mask = (freq >= f_lo) & (freq <= f_hi)
    bass_shape = (freq[bass_mask] - f_lo) / (f_hi - f_lo)
    bass_corr = correction[bass_mask]
    correction[bass_mask] = np.where(bass_corr > 0, bass_corr * bass_shape, bass_corr)
    correction[freq < f_lo] = 0.0

    hf_mask = freq >= 15000
    correction[hf_mask] = np.minimum(correction[hf_mask], 0.0)
    taper_mask = (freq >= 15000) & (freq <= 20000)
    if np.any(taper_mask):
        taper = 1 - (freq[taper_mask] - 15000) / 5000.0
        correction[taper_mask] *= taper
    correction[freq > 20000] = 0.0

    if len(boost_ceiling) != len(freq):
        raise ValueError("boost_ceiling must have the same length as freq.")
    upper = np.minimum(9.0, boost_ceiling)
    correction = np.clip(correction, -18, upper)
    eq_gain = np.interp(np.log10(GRAPHICEQ_FREQS), np.log10(freq), correction)
    return GRAPHICEQ_FREQS, eq_gain


def write_graphic_eq(path: Path, out_freq: np.ndarray, out_gain: np.ndarray) -> None:
    write_graphic_eq_file(path, out_freq, out_gain)
    print(f"  Written: {path}  ({len(out_freq)} bands)")


def save_fig(fig: plt.Figure, name: str) -> None:
    save_common_fig(fig, OUTPUT_DIR, name)


def main() -> None:
    args = parse_args()
    car_name = args.car
    slug = args.slug or car_name
    data_dir = BASE / car_name
    paths = build_measurement_files(data_dir, car_name, args.n)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading {car_name} L+R measurements...")
    require_files(paths)
    freq, spl_list = load_measurements(paths)

    raw_avg = energy_average(spl_list)
    raw_sm = variable_smooth(freq, raw_avg)
    ref_1k = float(raw_sm[np.argmin(np.abs(freq - 1000))])
    target = build_audiofrog_target(freq, ref_1k)
    plot_name = slug.upper()

    print(f"  Averaged {len(paths)} files")
    print(f"  1 kHz anchor: {ref_1k:.1f} dB")
    print("  Target: Audiofrog official target curve, normalized to the 1 kHz anchor")

    # Boost is capped by what the six positions agree on: where the
    # inter-position spread is small the deviation is common to every seat and
    # safe to boost; where it is large the boost would overshoot the seats
    # that don't share the dip.  (At/above 15 kHz boosting is banned outright
    # inside make_shared_eq, per the user's hearing-protection directive.)
    spread = position_spread(spl_list, freq)
    ceiling = consistency_boost_ceiling(spread, cap=9.0, tolerance=1.0, slope=1.5)
    eq_freq, eq_gain = make_shared_eq(freq.copy(), raw_avg.copy(), target.copy(), ceiling)

    ceiling_on_grid = np.interp(np.log10(GRAPHICEQ_FREQS), np.log10(freq), ceiling)
    ceiling_bites = int(np.sum((eq_gain > 0) & (eq_gain >= ceiling_on_grid - 0.05)))
    print(
        f"  Boost ceiling: spread median {np.median(spread):.1f} dB, "
        f"ceiling range [{ceiling.min():.1f}, {ceiling.max():.1f}] dB, "
        f"limiting at {ceiling_bites} of {len(GRAPHICEQ_FREQS)} bands"
    )

    write_graphic_eq(OUTPUT_DIR / f"{slug}_shared_eq.txt", eq_freq, eq_gain)

    gain_full = log_interp(freq, eq_freq, eq_gain)
    corrected = raw_sm + gain_full
    residual = corrected - target
    mask = (freq >= 100) & (freq <= 10000)
    rms = float(np.sqrt(np.mean(residual[mask] ** 2)))
    peak = float(np.max(np.abs(residual[mask])))
    print(f"  Residual vs target (100 Hz-10 kHz): RMS={rms:.2f} dB, peak={peak:.1f} dB")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(freq, raw_sm, label="Raw measurement (avg + variable smooth)", linewidth=1.4, alpha=0.65)
    ax.plot(freq, corrected, label="After EQ (predicted)", linewidth=2.0)
    ax.plot(freq, target, label="Target (Audiofrog)", linewidth=1.5, linestyle="--", color="red")
    ax.set_title(f"{plot_name} 90° L+R Shared EQ")
    setup_freq_axis(ax)
    save_fig(fig, f"{slug}_before_after.png")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axhline(0, color="red", linewidth=1, linestyle="--", label="Target")
    ax.axhspan(-3, 3, color="green", alpha=0.08, label="±3 dB zone")
    ax.plot(freq, residual, label="Residual after EQ", linewidth=1.5)
    ax.set_ylim(-15, 15)
    ax.set_title(f"{plot_name} EQ Residual vs Audiofrog Target")
    setup_freq_axis(ax, ylabel="Deviation from target (dB)")
    save_fig(fig, f"{slug}_residual.png")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
