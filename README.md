# Speaker Auto DSP

[![GitHub stars](https://img.shields.io/github/stars/BryleXia/speaker-auto-dsp)](https://github.com/BryleXia/speaker-auto-dsp/stargazers)
[![License: MIT](https://img.shields.io/github/license/BryleXia/speaker-auto-dsp)](LICENSE)
[![Python 3](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)]

[简体中文](README.zh-CN.md)

A small toolchain that turns [REW](https://www.roomeqwizard.com/) frequency-response
measurements into correction filters for [Equalizer APO](https://sourceforge.net/projects/equalizerapo/)
on Windows. Two independent pipelines are provided: one for stereo desktop/room
speakers, one for car audio.

| Before | After |
| --- | --- |
| ![Before](校准前.png) | ![After](效果.png) |

## Background

A loudspeaker's anechoic response is not what reaches the listener. Below roughly
200–300 Hz the room's modes dominate: standing waves add several dB at some
frequencies and cancel at others, and the pattern changes with position. Above
that region, boundary reflections and the speaker's own directivity shape the
response more gently but still measurably. A car cabin is the extreme case —
small volume, hard surfaces, and drivers mounted off-axis at unequal distances.

Three consequences drive the design of this tool:

- **A single measurement describes a single point in space.** Averaging several
  microphone positions around the head gives a response that is representative of
  listening rather than of one spot. The averaging is done on energy (power), not
  on dB values, which is what corresponds to summing acoustic power.
- **Not every dip should be filled.** A modal cancellation is a property of the
  sound field at that one position: a filter applies everywhere, so boosting
  overshoots at positions that do not have the null and barely changes the null
  itself, while consuming amplifier and driver headroom. A dip that appears
  consistently at every measured position is more likely a property of the
  speaker and can be corrected. This tool tells the two apart by the spread
  across positions, and limits boost accordingly.
- **The target is not a flat line.** A perceptually neutral steady-state in-room
  response carries a slight bass lift and a gentle downward tilt at high
  frequencies. The room pipeline uses a Harman-style nearfield target: a +3 dB
  low shelf at 105 Hz and −0.5 dB/octave above 1 kHz.

The shelf and tilt figures above are one adaptation, not a published standard
curve: the shape restates the general findings of listening-position preference
research, while the parameters were set for the nearfield desktop monitors this
tool was built around. If you are correcting a different kind of speaker in a
different kind of room — bookshelf speakers at a greater distance, floorstanders,
home theatre — substitute a target suited to that application. The curve lives
in a single function, `harman_nearfield_target` in `eq_common.py`, shared by
`eq_make.py` and `verify_eq.py` — swap it there. The bass boost ceilings in
`eq_make.py` likewise encode the limits of a small 3.5-inch woofer, not a
general rule.

Correction is applied as a **minimum-phase** filter, so the group delay stays low
and no pre-ringing is introduced.

## Requirements

```
pip install numpy matplotlib
```

REW for the measurements, Equalizer APO on the playback device.

## Room EQ

### 1. Measure

With APO bypassed, sweep each channel in REW at several microphone positions
around the listening spot. Export each sweep as a `.txt` file into `原始数据/`,
named `L …` for the left channel and `R …` for the right.

### 2. Generate

```
python eq_make.py
```

Outputs land in `output/`:

| File | Purpose |
| --- | --- |
| `room_eq_48000Hz.wav`, `room_eq_44100Hz.wav` | Stereo minimum-phase FIR kernels — the recommended output |
| `left_eq.txt`, `right_eq.txt` | GraphicEQ text, for setups where convolution is unavailable |
| `15…19_*.png` | Predicted before/after response, residual against target, and the boost-ceiling curves |

The script reads each generated `.wav` back and verifies its magnitude response
against the intended correction; the design keeps the worst-case error below
0.1 dB.

### 3. Apply

In the Equalizer APO configuration, one line:

```
Convolution: E:\…\output\room_eq_48000Hz.wav
```

The kernel's sample rate must match the playback device's sample rate, otherwise
APO will not load it — hence the two variants.

No `Preamp:` line is written. Overall gain and headroom are left to the user; the
script prints the largest boost it applied so an appropriate attenuation can be
chosen.

### 4. Verify

Re-measure with APO enabled, export into `挂载eq后测出来的数据/` using the same
naming convention, then:

```
python verify_eq.py
```

This plots the measured post-EQ response against the prediction and the target,
which is the only way to confirm that the filter did what it was designed to do.

## Car EQ

Car head units apply one shared curve rather than separate left and right
channels, so this pipeline averages all measurements together. Place the sweeps
for one vehicle in a directory named after it:

```
MyCar/
  L+R MyCar-1.txt
  L+R MyCar-2.txt
  ...
```

```
python car_eq.py --car MyCar                # → output/MyCar_shared_eq.txt
python car_eq.py --car MyCar --slug mycar   # custom output prefix
python car_eq.py --car MyCar --n 4          # use only the first 4 files
```

The target here is an Audiofrog in-car curve, and the output is written on a
fixed grid of 127 integer frequencies for compatibility with wavelet-style head
units. Vehicle directories are meant to be listed in `.gitignore`.

## License

[MIT](LICENSE) © 2026 BryleXia
