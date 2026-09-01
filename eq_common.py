from __future__ import annotations

import struct
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


FREQ_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# --- Input contract ----------------------------------------------------------
# The pipeline accepts exactly one input format: REW's default text export —
# unsmoothed, full-resolution linear FFT-bin grid starting at the first bin
# (~0.37 Hz).  Fractional-octave (ppo) exports are rejected: raw resolution
# is the measurement's ground truth, and the per-octave resample below is
# this pipeline's job, done once, deterministically, onto a 96-ppo log grid
# anchored at the first data point at or above 20 Hz — the same bin REW's
# own ppo export anchors to.
MIN_HZ = 20.0            # sub-audio bins in default exports are noise
CANON_PPO = 96           # canonical grid: 96 points per octave
MIN_SPAN_HZ = 5000.0     # guards against non-frequency-response exports
MIN_PTS_PER_OCT = 8      # guards against too-coarse source grids
GRID_MATCH_ATOL = 1e-3   # anchor-reconstruction drift, physically meaningless


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected a file, got: {path}")


def require_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required files:\n  {joined}")


def _canonicalize(freq: np.ndarray, spl: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Normalise a parsed REW grid (see the constants block for the contract).

    Only raw-resolution (linear FFT-bin) exports are accepted.  Fractional-
    octave exports are rejected outright — they are already-downsampled data,
    and the resample onto the 96-ppo grid is this pipeline's job, not REW's.
    The 96-ppo target grid is anchored at the first data point at or above
    20 Hz quantised to the source grid's step — REW anchors its own ppo
    exports to the same FFT bin, so the resampled grid lands on the same
    points REW's native 96-ppo export used.  The ceil rule for the grid end
    reproduces REW's behaviour of including the first ppo point at or beyond
    the data's end.
    """
    keep = freq >= MIN_HZ
    freq, spl = freq[keep], spl[keep]
    if len(freq) == 0:
        raise ValueError(f"No data at or above {MIN_HZ:.0f} Hz in {path}")
    if freq[-1] < MIN_SPAN_HZ:
        raise ValueError(
            f"{path.name}: data stops at {freq[-1]:.0f} Hz — expected a "
            f"full-range frequency-response export (reached {MIN_SPAN_HZ:.0f} Hz)"
        )
    if np.any(np.diff(freq) <= 0):
        raise ValueError(f"{path.name}: frequencies must be strictly increasing")

    log_diffs = np.diff(np.log2(freq))
    med = float(np.median(log_diffs))
    spread = float(np.max(np.abs(log_diffs - med))) / med
    if spread < 1e-4:
        ppo = 1.0 / med
        raise ValueError(
            f"{path.name}: this is a fractional-octave export ({ppo:.0f} ppo). "
            f"The pipeline only accepts REW's default export — raw resolution, "
            f"no smoothing (see README). Re-export with 'use native "
            f"resolution' and smoothing set to None."
        )

    # Irregular/linear source grid: check it is dense enough to resample.
    # The last edge reaches freq[-1] so the partial top octave is covered too.
    n_octs = int(np.floor(np.log2(freq[-1] / MIN_HZ)))
    edges = np.append(np.log2(MIN_HZ) + np.arange(n_octs + 1), np.log2(freq[-1]))
    counts, _ = np.histogram(np.log2(freq), bins=edges)
    if counts.size and counts.min() < MIN_PTS_PER_OCT:
        raise ValueError(
            f"{path.name}: source grid has fewer than {MIN_PTS_PER_OCT} points "
            f"per octave somewhere below {freq[-1]:.0f} Hz — export the "
            f"full-resolution measurement"
        )

    # Anchor reconstruction: the file's 6-decimal rounding puts ~1e-6 of
    # noise in every diff, and a median-step anchor drifts 55× that — which
    # the log grid then amplifies a thousandfold by 20 kHz.  On a uniform
    # grid the end-to-end span cancels the endpoint rounding, giving a step
    # accurate to ~1e-11.
    diffs = np.diff(freq)
    med_step = float(np.median(diffs))
    if float(np.max(np.abs(diffs - med_step))) < 0.01 * med_step:
        step = (freq[-1] - freq[0]) / (len(freq) - 1)
    else:
        step = med_step
    anchor = round(freq[0] / step) * step
    if anchor <= 0:
        raise ValueError(
            f"{path.name}: source grid too irregular to anchor a "
            f"{CANON_PPO}-ppo resample (first point {freq[0]:.3g} Hz, "
            f"median step {step:.3g} Hz)"
        )
    n = int(np.ceil(np.log2(freq[-1] / anchor) * CANON_PPO)) + 1
    grid = anchor * 2.0 ** (np.arange(n) / CANON_PPO)
    return grid, np.interp(np.log2(grid), np.log2(freq), spl)


def parse_rew(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a REW text export into (freq, spl) arrays.

    REW's header length varies with version and export options, so instead of
    skipping a fixed number of lines we keep every row whose first two
    whitespace-separated fields both parse as floats.  Comment banners, the
    column-title line, and blank lines are skipped automatically.

    The result is canonicalised (see :func:`_canonicalize`): sub-audio bins
    are dropped and the raw-resolution grid is resampled onto 96 ppo.  Only
    default exports (raw resolution, no smoothing) are accepted.
    """
    require_file(path)
    freqs: list[float] = []
    spls: list[float] = []
    with path.open(encoding="utf-8", errors="replace") as fp:
        for line in fp:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                freq_val = float(parts[0])
                spl_val = float(parts[1])
            except ValueError:
                continue
            freqs.append(freq_val)
            spls.append(spl_val)

    if not freqs:
        raise ValueError(f"Could not parse REW frequency/SPL data from {path}")

    freq = np.asarray(freqs, dtype=float)
    spl = np.asarray(spls, dtype=float)
    if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(spl)):
        raise ValueError(f"REW file contains non-finite values: {path}")
    if np.any(freq <= 0):
        raise ValueError(f"REW file contains non-positive frequencies: {path}")

    return _canonicalize(freq, spl, path)


def load_measurements(paths: list[Path]) -> tuple[np.ndarray, list[np.ndarray]]:
    require_files(paths)
    freq: np.ndarray | None = None
    spls: list[np.ndarray] = []
    for path in paths:
        current_freq, spl = parse_rew(path)
        if freq is None:
            freq = current_freq
        elif len(current_freq) != len(freq) or not np.allclose(
            current_freq, freq, rtol=0, atol=GRID_MATCH_ATOL
        ):
            raise ValueError(
                f"Frequency grid mismatch in {path} — all files in one batch "
                f"must share export settings (range and resolution)"
            )
        spls.append(spl)

    if freq is None or not spls:
        raise ValueError("No measurements were loaded.")
    return freq, spls


def energy_average(spl_list: list[np.ndarray]) -> np.ndarray:
    if not spl_list:
        raise ValueError("Cannot average an empty measurement list.")
    stack = np.vstack(spl_list)
    return 10 * np.log10(np.mean(10 ** (stack / 10), axis=0))


def smooth_oct(
    freq: np.ndarray,
    spl: np.ndarray,
    fraction: float = 1 / 6,
) -> np.ndarray:
    """Fractional-octave smoothing that adapts to the data's frequency spacing.

    Works correctly on any resolution — 48 PPO, 96 PPO, 384 PPO, or raw
    linearly-spaced FFT data.  The window width in *points* is derived from
    the actual local density of frequency points so that the smoothing
    bandwidth is always *fraction* octaves regardless of grid type.

    Edge effects are suppressed by reflect-padding the SPL array before
    convolution, then trimming back to the original length.

    Parameters
    ----------
    freq : 1-D array of positive frequencies (Hz), strictly increasing.
    spl  : 1-D array of SPL values (dB), same length as *freq*.
    fraction : smoothing bandwidth as a fraction of an octave (default 1/6).
    """
    if len(freq) != len(spl):
        raise ValueError("freq and spl must have the same length.")
    if len(spl) < 4:
        return spl.copy()
    if np.any(freq <= 0):
        raise ValueError("All frequencies must be positive for octave smoothing.")

    log_freq = np.log2(freq)
    log_diffs = np.diff(log_freq)

    # Median log2-spacing → median points-per-octave
    median_log_step = float(np.median(log_diffs))
    if median_log_step <= 0:
        raise ValueError("Frequencies must be strictly increasing.")
    ppo = 1.0 / median_log_step
    n_points = max(4, int(round(ppo * fraction)))

    # Cap window to data length
    n_points = min(n_points, len(spl))
    if n_points < 2:
        return spl.copy()

    # Reflect-pad to eliminate edge attenuation, then convolve, then trim.
    # We use mode="full" with an explicit start index rather than mode="same"
    # so the output alignment is obvious and independent of numpy's internal
    # centering convention (which varies between even/odd kernel lengths).
    pad = n_points // 2
    padded = np.pad(spl, pad, mode="reflect")
    window = np.hanning(n_points)
    window /= window.sum()
    full_conv = np.convolve(padded, window, mode="full")
    # full_conv has length len(padded) + n_points - 1.
    # The output aligned with spl[0] starts at index n_points - 1.
    return full_conv[n_points - 1 : n_points - 1 + len(spl)]


def estimate_ppo(freq: np.ndarray) -> float:
    """Estimate the median points-per-octave of a frequency grid."""
    log_diffs = np.diff(np.log2(freq))
    return 1.0 / float(np.median(log_diffs))


def position_spread(
    spl_list: list[np.ndarray],
    freq: np.ndarray,
    smoothing: float = 1 / 6,
    spread_smoothing: float = 1 / 3,
) -> np.ndarray:
    """Per-frequency standard deviation across several mic positions (dB).

    Each measurement is smoothed on its own first (so the spread reflects
    genuine position-to-position differences rather than measurement noise),
    then the standard deviation across positions is itself smoothed a little
    wider so the curve derived from it doesn't wobble point to point.

    A large spread means "this dip/peak moves when the listener moves" —
    a position-dependent modal null or reflection notch that no single EQ
    curve can fix.  A small spread means the deviation is the same everywhere
    and is therefore a property of the speaker or of stable boundary loading.
    """
    if len(spl_list) < 2:
        return np.zeros_like(freq)
    per_position = np.vstack([smooth_oct(freq, spl, smoothing) for spl in spl_list])
    return smooth_oct(freq, per_position.std(axis=0), spread_smoothing)


def physical_boost_ceiling(
    freq: np.ndarray, points: list[tuple[float, float]], cap: float
) -> np.ndarray:
    """Frequency-dependent boost ceiling from (Hz, dB) control points.

    Interpolated in log frequency; flat at the first/last value outside the
    control range.  Used to stop the correction from demanding output where
    the driver physically cannot deliver it (near and below the cabinet's
    port tuning, where excursion soars and efficiency collapses).
    """
    ctl_freq = np.asarray([p[0] for p in points], dtype=float)
    ctl_gain = np.asarray([p[1] for p in points], dtype=float)
    if np.any(np.diff(ctl_freq) <= 0):
        raise ValueError("Boost-ceiling control points must be sorted by frequency.")
    ceiling = np.interp(np.log10(freq), np.log10(ctl_freq), ctl_gain)
    return np.clip(ceiling, 0.0, cap)


def consistency_boost_ceiling(
    spread: np.ndarray, cap: float, tolerance: float = 1.0, slope: float = 1.5
) -> np.ndarray:
    """Boost ceiling driven by the inter-position spread from :func:`position_spread`.

    Spread up to *tolerance* dB costs nothing; every further dB of spread
    removes *slope* dB of the available boost.  This is the mechanism that
    keeps the correction from trying to fill position-dependent room nulls:
    boosting them overshoots at the positions that don't have the null and
    barely dents the ones that do.
    """
    return np.clip(cap - slope * np.maximum(0.0, spread - tolerance), 0.0, cap)


def shelf_db(freq: np.ndarray, fc: float, gain_db: float, q: float, fs: int = 48000) -> np.ndarray:
    """Exact low-shelf magnitude response in dB, based on the audio EQ cookbook."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    sq = 2 * np.sqrt(a) * alpha
    b0 = a * ((a + 1) - (a - 1) * cos_w0 + sq)
    b1 = 2 * a * ((a - 1) - (a + 1) * cos_w0)
    b2 = a * ((a + 1) - (a - 1) * cos_w0 - sq)
    a0 = (a + 1) + (a - 1) * cos_w0 + sq
    a1 = -2 * ((a - 1) + (a + 1) * cos_w0)
    a2 = (a + 1) + (a - 1) * cos_w0 - sq
    w = 2 * np.pi * freq / fs
    z = np.exp(1j * w)
    h = (b0 + b1 * z**-1 + b2 * z**-2) / (a0 + a1 * z**-1 + a2 * z**-2)
    return 20 * np.log10(np.maximum(np.abs(h), 1e-10))


def harman_nearfield_target(freq: np.ndarray, ref_level: float) -> np.ndarray:
    low_shelf = shelf_db(freq, fc=105, gain_db=3.0, q=0.65)
    hf_rolloff = np.where(freq > 1000, -0.5 * np.log2(freq / 1000), 0.0)
    return ref_level + low_shelf + hf_rolloff


def compute_correction(
    freq: np.ndarray,
    raw_spl: np.ndarray,
    target: np.ndarray,
    *,
    smoothing: float = 1 / 6,
    mid_band: tuple[float, float] = (200.0, 8000.0),
    level_offset: float | None = None,
    bass_taper: tuple[float, float] = (20.0, 65.0),
    hf_taper: tuple[float, float] = (18000.0, 20000.0),
    clip: tuple[float, float] = (-18.0, 9.0),
    boost_ceiling: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Full-resolution correction curve shared by the room scripts.

    Pipeline: fractional-octave smooth → ``target - smoothed`` → remove a
    level offset → taper the unreliable bass and ultrasonic regions to zero →
    clamp to a safe gain window.

    By default the removed offset is this channel's own mean over ``mid_band``.
    Pass an explicit ``level_offset`` to remove a *shared* offset instead — the
    room generator uses this so both channels are centred on the same level and
    the stereo image stays put (removing each channel's own offset would erase
    the very L/R level difference we want to equalise).

    ``bass_taper`` ramps the correction (both directions) to zero between its
    two frequencies and zeroes everything below the lower one.  Pass equal
    values — e.g. ``(40.0, 40.0)`` — for a hard floor with no ramp, which is
    what you want when a ``boost_ceiling`` already shapes the bass region and
    the ramp would only weaken the cuts there.

    ``boost_ceiling`` is an optional per-frequency upper limit (same length as
    *freq*) applied on top of ``clip[1]``; cuts are never restricted by it.
    See :func:`physical_boost_ceiling` and :func:`consistency_boost_ceiling`.

    Returns the full-resolution ``(correction, level_offset)``; downsample to
    GraphicEQ bands separately with :func:`downsample_bands`.
    """
    smoothed = smooth_oct(freq, raw_spl, smoothing)
    correction = target - smoothed

    if level_offset is None:
        lo, hi = mid_band
        mid_mask = (freq >= lo) & (freq <= hi)
        if not np.any(mid_mask):
            raise ValueError(
                f"No frequency points inside the mid-band {mid_band} Hz used for "
                f"level normalization (data spans {freq.min():.0f}–{freq.max():.0f} Hz)."
            )
        level_offset = float(np.mean(correction[mid_mask]))
    correction = correction - level_offset

    f_lo, f_hi = bass_taper
    if f_hi > f_lo:
        bass_mask = (freq >= f_lo) & (freq <= f_hi)
        correction[bass_mask] *= (freq[bass_mask] - f_lo) / (f_hi - f_lo)
    correction[freq < f_lo] = 0.0

    h_lo, h_hi = hf_taper
    taper_mask = (freq >= h_lo) & (freq <= h_hi)
    correction[taper_mask] *= 1 - (freq[taper_mask] - h_lo) / (h_hi - h_lo)
    correction[freq > h_hi] = 0.0

    upper: float | np.ndarray = clip[1]
    if boost_ceiling is not None:
        if len(boost_ceiling) != len(freq):
            raise ValueError("boost_ceiling must have the same length as freq.")
        upper = np.minimum(clip[1], boost_ceiling)
    correction = np.clip(correction, clip[0], upper)
    return correction, level_offset


def downsample_bands(
    freq: np.ndarray, values: np.ndarray, bands_per_oct: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample a full-resolution curve to roughly ``bands_per_oct`` per octave."""
    ppo = estimate_ppo(freq)
    step = max(1, int(round(ppo / bands_per_oct)))
    return freq[::step], values[::step]


def design_minimum_phase_ir(
    freq: np.ndarray,
    gain_db: np.ndarray,
    fs: int = 48000,
    n_fft: int = 65536,
    refine: bool = True,
) -> np.ndarray:
    """Design a minimum-phase FIR impulse response whose magnitude matches
    *gain_db* sampled at *freq* (Hz), for use with APO's Convolution command.

    Classic cepstral method (Oppenheim & Schafer): real cepstrum of the
    magnitude, causal fold, exponentiate back.  Outside the span of *freq*
    the response is flat 0 dB, which is correct here because the pipeline
    tapers the correction to zero at both ends of the measured grid.

    With ``refine`` (default) a second pass compensates the residual cepstral
    error by re-designing on (gain - measured error), which pushes the
    worst-case magnitude error well below 0.1 dB even at sharp room-mode cuts.
    """
    if len(freq) != len(gain_db):
        raise ValueError("freq and gain_db must have the same length.")
    if np.any(np.diff(freq) <= 0):
        raise ValueError("freq must be strictly increasing.")

    def build(gain: np.ndarray) -> np.ndarray:
        # Desired magnitude on the FFT bin grid (linear frequency, 0..Nyquist),
        # interpolated in log-frequency since EQ curves are log-spaced.
        bin_freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        inside = (bin_freq >= freq[0]) & (bin_freq <= freq[-1])
        mag_db = np.zeros_like(bin_freq)
        mag_db[inside] = np.interp(np.log10(bin_freq[inside]), np.log10(freq), gain)
        mag = 10 ** (mag_db / 20)

        # Real cepstrum of the (real, even) magnitude spectrum.
        log_mag_half = np.log(mag)                              # 0..N/2
        log_mag = np.concatenate([log_mag_half, log_mag_half[-2:0:-1]])
        cep = np.fft.ifft(log_mag).real

        # Causal (minimum-phase) fold of the cepstrum.
        cep_mp = np.zeros(n_fft)
        cep_mp[0] = cep[0]
        cep_mp[1 : n_fft // 2] = 2.0 * cep[1 : n_fft // 2]
        cep_mp[n_fft // 2] = cep[n_fft // 2]

        # Back to the complex spectrum, then to the time domain.
        return np.fft.ifft(np.exp(np.fft.fft(cep_mp))).real

    ir = build(gain_db)
    if not refine:
        return ir
    err = ir_magnitude_db(ir, freq, fs) - gain_db
    ir_refined = build(gain_db - err)
    err_refined = ir_magnitude_db(ir_refined, freq, fs) - gain_db
    return ir_refined if np.abs(err_refined).max() <= np.abs(err).max() else ir


def ir_magnitude_db(ir: np.ndarray, freq: np.ndarray, fs: int) -> np.ndarray:
    """Magnitude response (dB) of an IR, sampled at *freq* Hz."""
    spec = np.abs(np.fft.rfft(ir))
    bin_freq = np.fft.rfftfreq(len(ir), d=1.0 / fs)
    return 20 * np.log10(np.maximum(np.interp(freq, bin_freq, spec), 1e-10))


def write_stereo_ir_wav(path: Path, ir_l: np.ndarray, ir_r: np.ndarray, fs: int) -> float:
    """Write a stereo 32-bit float WAV impulse response for APO's Convolution.

    APO's official recommendation is 32-bit float (no clipping at values
    above 1.0), so the IR is written exactly as designed — no normalization,
    no baked-in level shift.  Overall output level stays the user's own
    adjustment.  Returns the IR peak amplitude (may exceed 1.0 where the
    correction boosts).
    """
    if len(ir_l) != len(ir_r):
        raise ValueError("ir_l and ir_r must have the same length.")
    interleaved = np.empty(len(ir_l) * 2, dtype="<f4")
    interleaved[0::2] = ir_l
    interleaved[1::2] = ir_r
    payload = interleaved.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(payload), b"WAVE",
        b"fmt ", 16, 3, 2, fs, fs * 8, 8, 32,
        b"data", len(payload),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        fp.write(header)
        fp.write(payload)
    return max(float(np.max(np.abs(ir_l))), float(np.max(np.abs(ir_r))))


def read_ir_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read back a float WAV written by :func:`write_stereo_ir_wav`.

    Minimal RIFF chunk-walking parser.  Returns ``(channels x samples
    array, sample rate)``.  Used to self-verify written IR files right
    after writing them.
    """
    raw = path.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {path}")
    fmt_body = None
    data = None
    pos = 12
    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        chunk_size = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        body = raw[pos + 8 : pos + 8 + chunk_size]
        if chunk_id == b"fmt ":
            audio_format, channels, rate, _byte_rate, _align, bits = struct.unpack(
                "<HHIIHH", body[:16]
            )
            if audio_format != 3 or bits != 32:
                raise ValueError(
                    f"Expected 32-bit float WAV, got format {audio_format}/{bits}-bit in {path}"
                )
            fmt_body = (channels, rate)
        elif chunk_id == b"data":
            data = body
        pos += 8 + chunk_size + (chunk_size & 1)  # chunks are word-aligned
    if fmt_body is None or data is None:
        raise ValueError(f"fmt/data chunk missing in {path}")
    channels, rate = fmt_body
    samples = np.frombuffer(data, dtype="<f4").reshape(-1, channels)
    return samples.T.copy(), rate


def parse_graphic_eq(path: Path) -> tuple[np.ndarray, np.ndarray]:
    require_file(path)
    text = path.read_text(encoding="utf-8").strip()
    line = ""
    for candidate in text.splitlines():
        if candidate.strip().lower().startswith("graphiceq:"):
            line = candidate.strip()[len("graphiceq:") :].strip()
            break
    if not line and text.lower().startswith("graphiceq:"):
        line = text[len("graphiceq:") :].strip()
    if not line:
        raise ValueError(f"GraphicEQ line not found in {path}")

    freqs: list[float] = []
    gains: list[float] = []
    for pair in line.split(";"):
        parts = pair.strip().split()
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError(f"Invalid GraphicEQ pair in {path}: {pair!r}")
        freqs.append(float(parts[0]))
        gains.append(float(parts[1]))

    if not freqs:
        raise ValueError(f"GraphicEQ file contained no bands: {path}")
    freq_arr = np.asarray(freqs, dtype=float)
    gain_arr = np.asarray(gains, dtype=float)
    if np.any(freq_arr <= 0):
        raise ValueError(f"GraphicEQ frequencies must be positive: {path}")
    if np.any(np.diff(freq_arr) < 0):
        raise ValueError(f"GraphicEQ frequencies must be sorted: {path}")

    if np.any(np.diff(freq_arr) == 0):
        unique_freqs: list[float] = []
        unique_gains: list[float] = []
        for freq, gain in zip(freq_arr, gain_arr):
            if unique_freqs and freq == unique_freqs[-1]:
                unique_gains[-1] = float(gain)
            else:
                unique_freqs.append(float(freq))
                unique_gains.append(float(gain))
        freq_arr = np.asarray(unique_freqs, dtype=float)
        gain_arr = np.asarray(unique_gains, dtype=float)

    return freq_arr, gain_arr


def _format_band_freq(freq: float) -> str:
    """Band frequency with up to 2 decimals, trailing zeros trimmed.

    APO accepts decimal frequencies; dense band tables need them, otherwise
    adjacent points below ~50 Hz collide when rounded to integers
    (20.14 and 20.43 would both become "20").
    """
    return f"{freq:.2f}".rstrip("0").rstrip(".") or "0"


def write_graphic_eq(path: Path, out_freq: np.ndarray, out_gain: np.ndarray, preamp: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = "; ".join(
        f"{_format_band_freq(freq)} {gain:.2f}" for freq, gain in zip(out_freq, out_gain)
    )
    with path.open("w", encoding="utf-8") as fp:
        if preamp is not None:
            fp.write(f"Preamp: {preamp:.1f} dB\n")
        fp.write(f"GraphicEQ: {parts}\n")


def log_interp(freq: np.ndarray, src_freq: np.ndarray, src_value: np.ndarray) -> np.ndarray:
    if np.any(freq <= 0) or np.any(src_freq <= 0):
        raise ValueError("Log interpolation requires positive frequencies.")
    return np.interp(np.log10(freq), np.log10(src_freq), src_value)


def residual_stats(signal: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if not np.any(mask):
        raise ValueError("Statistics mask selected no frequency points.")
    residual = signal[mask] - reference[mask]
    return float(np.sqrt(np.mean(residual**2))), float(np.max(np.abs(residual)))


def setup_freq_axis(ax: plt.Axes, with_legend: bool = True, ylabel: str = "SPL (dB)") -> None:
    ax.set_xscale("log")
    ax.set_xlim(20, 20000)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.0f}" if x < 1000 else f"{x / 1000:.0f}k")
    )
    ax.xaxis.set_major_locator(ticker.FixedLocator(FREQ_TICKS))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    if with_legend:
        ax.legend(fontsize=9)


def save_fig(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path
