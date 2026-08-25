"""
Verify EQ by comparing measured response, predicted response, and target.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eq_common import (
    energy_average,
    harman_nearfield_target,
    load_measurements,
    log_interp,
    parse_graphic_eq,
    parse_rew,
    residual_stats,
    save_fig,
    setup_freq_axis,
    smooth_oct,
)


BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"

ORIG_DIR = BASE / "原始数据"
MEAS_DIR = BASE / "挂载eq后测出来的数据"

EQ_FILES = {
    "L": OUTPUT_DIR / "left_eq.txt",
    "R": OUTPUT_DIR / "right_eq.txt",
}

STAT_F_LO = 100
STAT_F_HI = 10000


def _discover_orig_files(data_dir: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for ch, pat in [("L", "L *.txt"), ("R", "R *.txt")]:
        found = sorted(data_dir.glob(pat))
        if not found:
            raise FileNotFoundError(f"No {ch}-channel raw files matching '{pat}' in {data_dir}")
        result[ch] = found
    return result


def _discover_meas_files(meas_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for ch, pat in [("L", "L *.txt"), ("R", "R *.txt")]:
        found = sorted(meas_dir.glob(pat))
        if not found:
            raise FileNotFoundError(f"No {ch}-channel with-EQ file found in {meas_dir}")
        result[ch] = found[0]
    return result


def main() -> None:
    print("Loading raw (bypass) measurements...")
    orig_files = _discover_orig_files(ORIG_DIR)
    freq: np.ndarray | None = None
    orig_spl: dict[str, np.ndarray] = {}
    for ch, paths in orig_files.items():
        freq_ch, spls = load_measurements(paths)
        if freq is None:
            freq = freq_ch
        elif len(freq_ch) != len(freq) or not np.allclose(freq_ch, freq, rtol=0, atol=1e-6):
            raise ValueError(f"Frequency grid mismatch in {ch} raw measurements.")
        orig_spl[ch] = energy_average(spls)

    if freq is None:
        raise ValueError("No raw measurements were loaded.")

    orig_sm = {ch: smooth_oct(freq, orig_spl[ch]) for ch in ("L", "R")}
    # Fit target to mid-band mean (200-8000 Hz) so it's centred on the measurements
    mid_mask = (freq >= 200) & (freq <= 8000)
    avg_sm = (orig_sm["L"] + orig_sm["R"]) / 2
    harman_shape = harman_nearfield_target(freq, 0.0)  # target shape at ref_level=0
    ref_level = float(np.mean(avg_sm[mid_mask] - harman_shape[mid_mask]))
    target = harman_nearfield_target(freq, ref_level)

    print("Loading with-EQ measurements...")
    meas_files = _discover_meas_files(MEAS_DIR)
    meas_spl: dict[str, np.ndarray] = {}
    for ch in ("L", "R"):
        meas_freq, spl = parse_rew(meas_files[ch])
        if len(meas_freq) != len(freq) or not np.allclose(meas_freq, freq, rtol=0, atol=1e-6):
            raise ValueError(f"Frequency grid mismatch in {ch} with-EQ measurement.")
        meas_spl[ch] = smooth_oct(meas_freq, spl)

    print("Loading GraphicEQ configs and computing prediction...")
    predicted: dict[str, np.ndarray] = {}
    for ch in ("L", "R"):
        eq_freq, eq_gain = parse_graphic_eq(EQ_FILES[ch])
        predicted[ch] = orig_sm[ch] + log_interp(freq, eq_freq, eq_gain)

    stat_mask = (freq >= STAT_F_LO) & (freq <= STAT_F_HI)
    print()
    print("=" * 60)
    print(f"Residual Statistics ({STAT_F_LO}-{STAT_F_HI} Hz)")
    print("=" * 60)

    for ch in ("L", "R"):
        rms_meas_tgt, pk_meas_tgt = residual_stats(meas_spl[ch], target, stat_mask)
        rms_pred_tgt, pk_pred_tgt = residual_stats(predicted[ch], target, stat_mask)
        rms_meas_pred, pk_meas_pred = residual_stats(meas_spl[ch], predicted[ch], stat_mask)
        print()
        print(f"  {ch} Channel:")
        print(f"    Measured vs Target:    RMS={rms_meas_tgt:.2f} dB  peak={pk_meas_tgt:.1f} dB")
        print(f"    Predicted vs Target:   RMS={rms_pred_tgt:.2f} dB  peak={pk_pred_tgt:.1f} dB")
        print(f"    Measured vs Predicted: RMS={rms_meas_pred:.2f} dB  peak={pk_meas_pred:.1f} dB")

    print("\nGenerating plots...")
    ch_colors = {"L": "steelblue", "R": "darkorange"}
    for ch in ("L", "R"):
        color = ch_colors[ch]
        rms_mt, pk_mt = residual_stats(meas_spl[ch], target, stat_mask)
        rms_pt, pk_pt = residual_stats(predicted[ch], target, stat_mask)

        # --- Plot 1: Measured vs Predicted vs Target ---
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(freq, meas_spl[ch], label="Measured (with EQ)", linewidth=2.0, color=color)
        ax.plot(freq, predicted[ch], label="Predicted (raw + EQ)", linewidth=1.5, color=color, linestyle="--")
        ax.plot(freq, target, label="Target (Harman-Nearfield)", linewidth=1.5, linestyle=":", color="red")
        ax.set_title(
            f"{ch} Channel: Measured vs Predicted vs Target\n"
            f"Meas vs Target: RMS={rms_mt:.2f} dB, peak={pk_mt:.1f} dB  |  "
            f"Pred vs Target: RMS={rms_pt:.2f} dB, peak={pk_pt:.1f} dB"
        )
        ax.set_ylim(50, 100)
        setup_freq_axis(ax)
        save_fig(fig, OUTPUT_DIR, f"V_{ch}_verify.png")

        # --- Plot 2: Raw (before EQ) vs Measured (after EQ) vs Target ---
        rms_raw_tgt, pk_raw_tgt = residual_stats(orig_sm[ch], target, stat_mask)
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(freq, orig_sm[ch], label="Raw (before EQ)", linewidth=1.5, color="gray", linestyle="--")
        ax2.plot(freq, meas_spl[ch], label="Measured (after EQ)", linewidth=2.0, color=color)
        ax2.plot(freq, target, label="Target (Harman-Nearfield)", linewidth=1.5, linestyle=":", color="red")
        ax2.set_title(
            f"{ch} Channel: Raw vs EQ'd vs Target\n"
            f"Raw vs Target: RMS={rms_raw_tgt:.2f} dB, peak={pk_raw_tgt:.1f} dB  |  "
            f"EQ'd vs Target: RMS={rms_mt:.2f} dB, peak={pk_mt:.1f} dB"
        )
        ax2.set_ylim(50, 100)
        setup_freq_axis(ax2)
        save_fig(fig2, OUTPUT_DIR, f"V_{ch}_raw_vs_eq.png")

    # --- Combined L/R: Measured vs Predicted vs Target ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for ch, color in ch_colors.items():
        ax.plot(freq, meas_spl[ch], label=f"{ch} Measured", linewidth=2.0, color=color)
        ax.plot(freq, predicted[ch], label=f"{ch} Predicted", linewidth=1.5, color=color, linestyle="--")
    ax.plot(freq, target, label="Target (Harman-Nearfield)", linewidth=1.5, linestyle=":", color="red")
    diff = meas_spl["L"] - meas_spl["R"]
    rms_lr = float(np.sqrt(np.mean(diff[stat_mask] ** 2)))
    ax.set_title(f"L/R Measured vs Predicted vs Target  (L-R Meas RMS: {rms_lr:.2f} dB)")
    ax.set_ylim(50, 100)
    setup_freq_axis(ax)
    save_fig(fig, OUTPUT_DIR, "V_LR_verify.png")

    # --- Combined L/R: Raw vs EQ'd vs Target ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for ch, color in ch_colors.items():
        ax.plot(freq, orig_sm[ch], label=f"{ch} Raw", linewidth=1.5, color=color, linestyle="--")
        ax.plot(freq, meas_spl[ch], label=f"{ch} EQ'd", linewidth=2.0, color=color)
    ax.plot(freq, target, label="Target (Harman-Nearfield)", linewidth=1.5, linestyle=":", color="red")
    diff_raw = orig_sm["L"] - orig_sm["R"]
    rms_lr_raw = float(np.sqrt(np.mean(diff_raw[stat_mask] ** 2)))
    diff_meas = meas_spl["L"] - meas_spl["R"]
    rms_lr_meas = float(np.sqrt(np.mean(diff_meas[stat_mask] ** 2)))
    ax.set_title(
        f"L/R Raw vs EQ'd vs Target  "
        f"(L-R Raw RMS: {rms_lr_raw:.2f} dB, L-R EQ'd RMS: {rms_lr_meas:.2f} dB)"
    )
    ax.set_ylim(50, 100)
    setup_freq_axis(ax)
    save_fig(fig, OUTPUT_DIR, "V_LR_raw_vs_eq.png")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
