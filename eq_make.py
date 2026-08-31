"""
eq_make.py — Daily-use EQ generator.

Reads listening-position measurements from 原始数据/, computes GraphicEQ
corrections for L and R channels, and writes output/left_eq.txt and
output/right_eq.txt ready to paste into Equalizer APO.

Includes L/R broadband level balance compensation so that asymmetric
speaker placement doesn't cause the soundstage to drift off-centre.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eq_common import (
    GRID_MATCH_ATOL,
    compute_correction,
    consistency_boost_ceiling,
    design_minimum_phase_ir,
    energy_average,
    harman_nearfield_target,
    ir_magnitude_db,
    load_measurements,
    log_interp,
    physical_boost_ceiling,
    position_spread,
    read_ir_wav,
    residual_stats,
    save_fig,
    setup_freq_axis,
    smooth_oct,
    write_graphic_eq as write_graphic_eq_file,
    write_stereo_ir_wav,
)


BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "output"
ORIG_DIR = BASE / "原始数据"

# --- Boost limiting ------------------------------------------------------
# Cuts are never limited beyond the -18 dB floor: removing energy is always
# safe and usually the single most audible improvement.  Boost is limited by
# two independent ceilings, and the smaller of the two wins.
#
# 1. PHYSICAL_CEILING_POINTS — what the cabinet can actually deliver.  The
#    near-field (room-free) measurements in 近场数据/ bottom out at 55-60 Hz,
#    the signature of a bass-reflex box unloading around its port tuning:
#    normalized to 200 Hz-2 kHz the speaker itself reads +2.1 dB at 100 Hz,
#    -3.9 at 70 Hz and -13.5 at 55 Hz.  The apparent "hole" at 45-58 Hz in the
#    listening-position curve is mostly the contrast with 63-80 Hz, which the
#    room lifts by +10 to +12 dB.  Asking a 3.5-inch driver on 18 W for +6 dB
#    at 55 Hz is 4x the power and ~2x the excursion right where it is least
#    efficient — audible distortion, no extra output.
# 2. SPREAD_* — what is actually correctable.  Deviations that are identical
#    at all 8 head positions are a property of the speaker/boundary and can be
#    equalized; deviations that move with the listener cannot.  At 110 Hz the
#    eight L-channel positions read 64.7 to 81.0 dB (sd 5.6) — a textbook
#    position-dependent modal null.  At 55 Hz they span 1.1 dB (sd 0.4).
BOOST_CAP = 6.0
PHYSICAL_CEILING_POINTS = [(20.0, 0.0), (40.0, 0.0), (70.0, 2.0), (100.0, 4.0), (150.0, 6.0)]
SPREAD_TOLERANCE = 1.0
SPREAD_SLOPE = 1.5

# Everything below this is zeroed outright: the speaker itself rolls off at
# roughly 24 dB/oct under ~32 Hz and the mic reads its own noise floor there.
BASS_FLOOR = 40.0

# Convolution IR parameters.  65536 taps (~1.4 s): bin spacing under 1 Hz,
# and the design includes one refinement pass — residual magnitude error
# stays well below 0.1 dB even at the sharpest room-mode cuts.  APO requires
# the IR sample rate to match the playback device exactly (mismatch makes
# the convolution fail to load), so one file is written per common rate.
IR_N = 65536
IR_FS_LIST = (48000, 44100)


def _discover_orig_files(data_dir: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for ch, pat in [("L", "L *.txt"), ("R", "R *.txt")]:
        found = sorted(data_dir.glob(pat))
        if not found:
            raise FileNotFoundError(f"No {ch}-channel files matching '{pat}' in {data_dir}")
        on_axis = [f for f in found if "90" not in f.stem]
        if on_axis and len(on_axis) != len(found):
            print(
                f"  WARNING: {data_dir} mixes {len(on_axis)} on-axis file(s) (no '90' "
                f"in the name) with {len(found) - len(on_axis)} listening-position "
                f"file(s) for {ch}; they will all be averaged together. "
                "eq_make expects listening-position (90-degree) measurements only."
            )
        result[ch] = found
    return result


def make_eq(
    freq: np.ndarray,
    raw_spl: np.ndarray,
    target: np.ndarray,
    boost_ceiling: np.ndarray,
    level_offset: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute correction = target - smooth(raw_spl), then remove a level offset,
    zero the unusable bass, and clamp gain against a per-frequency ceiling.

    With ``level_offset=None`` the channel's own mid-band offset is removed and
    returned (used to probe each channel's natural level).  Pass a shared
    offset to centre both channels identically — see :func:`main`.
    """
    # bass_taper with equal endpoints = hard floor, no ramp.  The old
    # 40-65 Hz ramp is now redundant: PHYSICAL_CEILING_POINTS shapes that
    # region with a physically motivated curve instead, and unlike the ramp it
    # does not weaken cuts (a 45 Hz peak, if one ever shows up, should be cut
    # in full).  Overall output level is deliberately NOT written — the user
    # sets Preamp/headroom themselves — so the cap guards the analog side
    # only: the 18 W amp and the 3.5-inch driver.
    correction, used_offset = compute_correction(
        freq,
        raw_spl,
        target,
        level_offset=level_offset,
        bass_taper=(BASS_FLOOR, BASS_FLOOR),
        clip=(-18.0, BOOST_CAP),
        boost_ceiling=boost_ceiling,
    )
    # Full-resolution output: every point of the measurement grid becomes a
    # band point.  APO's GraphicEQ is FFT-based and handles ~1000-point tables
    # fine, and skipping downsampling means zero interpolation loss — a
    # 12-bands/oct table measurably under-cut sharp room-mode notches by up to
    # ~2.3 dB because APO interpolates linearly between band points.
    # car_eq.py uses its own fixed grid.
    out_freq, out_gain = freq, correction
    return out_freq, out_gain, used_offset


def write_graphic_eq(path: Path, out_freq: np.ndarray, out_gain: np.ndarray) -> None:
    # No Preamp line: overall level is the user's own adjustment.  Print the
    # max boost so they know how much headroom to dial in themselves.
    write_graphic_eq_file(path, out_freq, out_gain)
    max_boost = float(np.max(out_gain))
    print(
        f"  Written: {path}  ({len(out_freq)} bands, "
        f"max boost={max_boost:+.1f} dB - Preamp is yours to set)"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading original measurements (APO bypassed)...")
    orig_files = _discover_orig_files(ORIG_DIR)
    orig_avg: dict[str, np.ndarray] = {}
    orig_spread: dict[str, np.ndarray] = {}
    freq: np.ndarray | None = None
    for ch, paths in orig_files.items():
        freq_ch, spls = load_measurements(paths)
        if freq is None:
            freq = freq_ch
        elif len(freq_ch) != len(freq) or not np.allclose(
            freq_ch, freq, rtol=0, atol=GRID_MATCH_ATOL
        ):
            raise ValueError(f"Frequency grid mismatch in {ch} measurements.")
        orig_avg[ch] = energy_average(spls)
        orig_spread[ch] = position_spread(spls, freq_ch)
        print(f"  {ch}: {len(paths)} positions, inter-position spread "
              f"median {np.median(orig_spread[ch]):.1f} dB, max {orig_spread[ch].max():.1f} dB")

    if freq is None:
        raise ValueError("No room measurements were loaded.")

    orig_sm = {ch: smooth_oct(freq, orig_avg[ch]) for ch in ("L", "R")}
    idx_1k = int(np.argmin(np.abs(freq - 1000)))
    common_ref = float((orig_sm["L"][idx_1k] + orig_sm["R"][idx_1k]) / 2)
    target = harman_nearfield_target(freq, common_ref)
    print(f"  Common ref (1kHz avg): {common_ref:.1f} dB")
    for ch in ("L", "R"):
        print(f"  {ch}: 1kHz = {orig_sm[ch][idx_1k]:.1f} dB")

    mid = (freq >= 200) & (freq <= 8000)
    lr_level_diff = float(np.mean(orig_sm["L"][mid]) - np.mean(orig_sm["R"][mid]))

    print("\nComputing EQ corrections (1/6-oct smooth, dual boost ceiling)...")
    physical = physical_boost_ceiling(freq, PHYSICAL_CEILING_POINTS, BOOST_CAP)
    ceiling: dict[str, np.ndarray] = {}
    for ch in ("L", "R"):
        consistency = consistency_boost_ceiling(
            orig_spread[ch], BOOST_CAP, SPREAD_TOLERANCE, SPREAD_SLOPE
        )
        ceiling[ch] = np.minimum(physical, consistency)
    for label, lo, hi in [("45-70 Hz", 45, 70), ("70-150 Hz", 70, 150), ("150 Hz+", 150, 20000)]:
        band = (freq >= lo) & (freq <= hi)
        print(
            f"  boost ceiling {label:>10}: physical {physical[band].min():+.1f}..{physical[band].max():+.1f} dB   "
            + "   ".join(f"{ch} final {ceiling[ch][band].min():+.1f}..{ceiling[ch][band].max():+.1f}" for ch in ("L", "R"))
        )

    # First pass: each channel's natural mid-band offset (correction discarded).
    natural_offsets = {ch: make_eq(freq, orig_avg[ch], target, ceiling[ch])[2] for ch in ("L", "R")}

    # Centre both channels on a *shared* offset so that, after EQ, L and R land
    # on the same midrange level — the stereo image stays centred.  Removing
    # each channel's own offset would instead erase the L/R level difference we
    # actually want to correct.
    common_offset = (natural_offsets["L"] + natural_offsets["R"]) / 2.0

    eq_freq: dict[str, np.ndarray] = {}
    eq_gain: dict[str, np.ndarray] = {}
    for ch in ("L", "R"):
        eq_freq[ch], eq_gain[ch], _ = make_eq(
            freq, orig_avg[ch], target, ceiling[ch], level_offset=common_offset
        )
        rms = float(np.sqrt(np.mean(eq_gain[ch] ** 2)))
        limited = (eq_gain[ch] > 0) & (eq_gain[ch] >= ceiling[ch] - 1e-9)
        print(
            f"  {ch}: gain [{eq_gain[ch].min():+.1f}, {eq_gain[ch].max():+.1f}] dB  "
            f"RMS={rms:.1f} dB  ceiling-limited at {limited.sum()} of {len(freq)} points"
        )
    print(
        f"\n  Measured L-R midrange level: {lr_level_diff:+.2f} dB "
        f"(positive = L louder) - corrected to centre via shared offset {common_offset:+.2f} dB"
    )

    print("\nWriting EQ configs...")
    write_graphic_eq(OUTPUT_DIR / "left_eq.txt", eq_freq["L"], eq_gain["L"])
    write_graphic_eq(OUTPUT_DIR / "right_eq.txt", eq_freq["R"], eq_gain["R"])

    # True-convolution output: stereo minimum-phase FIRs implementing the
    # full-resolution correction exactly, one WAV per sample rate, loaded in
    # APO with a single `Convolution:` line.  32-bit float and un-normalized:
    # the IR is the correction itself, and overall level stays the user's own
    # adjustment.  (GraphicEQ text remains as fallback; the car pipeline needs
    # it for Wavelet.)
    print(f"\nDesigning minimum-phase convolution IRs ({IR_N} taps, refined)...")
    for ir_fs in IR_FS_LIST:
        irs = {
            ch: design_minimum_phase_ir(freq, eq_gain[ch], fs=ir_fs, n_fft=IR_N)
            for ch in ("L", "R")
        }
        ir_path = OUTPUT_DIR / f"room_eq_{ir_fs}Hz.wav"
        peak = write_stereo_ir_wav(ir_path, irs["L"], irs["R"], fs=ir_fs)
        # Read the file back from disk and verify its magnitude really is
        # the correction curve — catches any WAV-writing mistake right here.
        wav_channels, wav_fs = read_ir_wav(ir_path)
        for idx, ch in enumerate(("L", "R")):
            err = ir_magnitude_db(wav_channels[idx], freq, fs=wav_fs) - eq_gain[ch]
            print(
                f"  {ir_fs} Hz {ch}: read-back vs correction  "
                f"max={np.abs(err).max():.3f} dB  RMS={np.sqrt(np.mean(err ** 2)):.4f} dB"
            )
        print(
            f"  Written: {ir_path}  (stereo 32-bit float, IR peak {peak:.2f}, "
            "no normalization - level is yours)"
        )

    predicted: dict[str, np.ndarray] = {}
    mask = (freq >= 100) & (freq <= 10000)
    for ch in ("L", "R"):
        predicted[ch] = orig_sm[ch] + log_interp(freq, eq_freq[ch], eq_gain[ch])

    print("\nGenerating plots...")
    for ch, color in [("L", "steelblue"), ("R", "darkorange")]:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(freq, orig_sm[ch], label="Raw measurement (avg)", linewidth=1.5, color=color, alpha=0.55)
        ax.plot(freq, predicted[ch], label="Predicted after EQ", linewidth=2.0, color=color)
        ax.plot(freq, target, label="Target (Harman-Nearfield)", linewidth=1.5, linestyle=":", color="red")
        rms, _ = residual_stats(predicted[ch], target, mask)
        ax.set_title(f"{ch} Channel: EQ prediction  (RMS vs target: {rms:.2f} dB)")
        ax.set_ylim(50, 100)
        setup_freq_axis(ax)
        save_fig(fig, OUTPUT_DIR, f"15_{ch}_fresh_eq.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    for ch, color in [("L", "steelblue"), ("R", "darkorange")]:
        ax.plot(freq, predicted[ch], label=f"{ch} predicted", linewidth=2, color=color)
    ax.plot(freq, target, label="Target (Harman-Nearfield)", linewidth=1.5, linestyle=":", color="red")
    diff = predicted["L"] - predicted["R"]
    rms_lr = float(np.sqrt(np.mean(diff[mask] ** 2)))
    ax.set_title(f"L vs R - EQ predicted  (L-R RMS: {rms_lr:.2f} dB)")
    ax.set_ylim(50, 100)
    setup_freq_axis(ax)
    save_fig(fig, OUTPUT_DIR, "16_LR_fresh_eq.png")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axhline(0, color="red", linewidth=1, linestyle="--", label="Target")
    ax.axhspan(-3, 3, color="green", alpha=0.08, label="+/-3 dB zone")
    for ch, color in [("L", "steelblue"), ("R", "darkorange")]:
        ax.plot(freq, predicted[ch] - target, label=f"{ch} predicted residual", linewidth=1.5, color=color)
    ax.set_ylim(-10, 10)
    ax.set_title("EQ - Predicted Residual vs Harman-Nearfield Target")
    setup_freq_axis(ax, ylabel="Deviation from target (dB)")
    save_fig(fig, OUTPUT_DIR, "17_residual_fresh_eq.png")

    # What the measurement actually looks like before any EQ, against the
    # target — the reference picture for judging whether the target itself
    # (rather than the correction) is what needs adjusting.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for ch, color in [("L", "steelblue"), ("R", "darkorange")]:
        ax.plot(freq, orig_avg[ch], linewidth=0.9, color=color, alpha=0.35,
                label=f"{ch} energy-avg (raw, {len(orig_files[ch])} files)")
        ax.plot(freq, orig_sm[ch], linewidth=2.0, color=color, label=f"{ch} avg + 1/6-oct smooth")
    ax.plot(freq, target, linewidth=1.5, linestyle=":", color="red", label="Target (Harman-Nearfield)")
    ax.axvline(1000, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(1050, 55, f"1 kHz anchor = {common_ref:.1f} dB", fontsize=9, color="gray")
    ax.set_ylim(35, 100)
    ax.set_title("Energy-Averaged L/R (raw vs 1/6-oct smoothed) vs Target")
    setup_freq_axis(ax)
    save_fig(fig, OUTPUT_DIR, "18_avg_LR_vs_target.png")

    # Where each boost ceiling bites: raw demand vs the ceiling vs what was
    # actually applied.  Any gap between the dashed demand line and the solid
    # correction is boost we deliberately declined to ask for.
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, ch, color in zip(axes, ("L", "R"), ("steelblue", "darkorange")):
        demand = target - orig_sm[ch] - common_offset
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.plot(freq, demand, linewidth=1.0, color="gray", linestyle="--", label="Raw demand (target - measured)")
        ax.plot(freq, physical, linewidth=1.2, color="green", linestyle=":", label="Physical ceiling")
        ax.plot(freq, ceiling[ch], linewidth=1.5, color="red", label="Final ceiling (physical & consistency)")
        ax.plot(freq, eq_gain[ch], linewidth=2.0, color=color, label=f"{ch} applied correction")
        ax.plot(freq, orig_spread[ch], linewidth=1.0, color="purple", alpha=0.6, label="Inter-position spread (sd)")
        ax.set_ylim(-20, 14)
        ax.set_title(f"{ch} - boost ceiling vs demand")
        setup_freq_axis(ax, ylabel="Gain (dB)")
    axes[0].set_xlabel("")
    save_fig(fig, OUTPUT_DIR, "19_boost_ceiling.png")

    print("\n" + "=" * 60)
    print("EQ Summary")
    print("=" * 60)
    for ch in ("L", "R"):
        rms, peak = residual_stats(predicted[ch], target, mask)
        print(f"  {ch}: predicted residual (100Hz-10kHz)  RMS={rms:.2f} dB   peak={peak:.1f} dB")
    print(f"\n  Shared offset removed: {common_offset:+.2f} dB")
    print(f"  Measured L-R midrange level diff: {lr_level_diff:+.2f} dB (corrected to centre)")
    print(f"  L/R predicted difference: {rms_lr:.2f} dB RMS")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
