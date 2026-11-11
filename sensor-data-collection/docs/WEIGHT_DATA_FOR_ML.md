# Dead-weight tactile dataset: how to train on it

For: a teammate who wants to train a model (neural net, GBM, anything) that
tells **which pad** of the 4x4 Velostat board carries a weight and **how many
grams** it is, from one 4x4 frame of sensor readings.

## 1. Where the data is

Everything lives under `C:\tactile_data\weight_cal\` (not in git; ask for a
copy). One folder per calibration session, one pad per session:

```
capture_20261031_202130_R1C1\        pad R1C1, 5 weights x 3 repetitions
    sensor_raw.csv                   every frame: host_received_ns, frame_index, device_millis,
                                     r1c1_adc12 ... r4c4_adc12  (16 raw 12-bit ADC values, 0-4095)
    session.json                     board facts: pad_under_test, adc_bits, pressure_lowers_adc, ...
    weights\events.csv               the operator's key presses: step, mass_g, on/off, host time
    force_alignment\frames.csv       per-frame labels: load_N, press (step), phase
    report\                          per-session report (weight_report.py)
capture_20261031_203125_R1C2\        pad R1C2 ...
capture_20261031_204118_R1C3\
capture_20261031_205246_R1C4\
```

`C:\tactile_data\weight_cal_old\` holds what was moved out: three aborted
trial runs, an earlier 2-repetition R1C1 run, and `capture_20261031_195035`
on R2C2, a pad that was overloaded to 20 N on the Mark-10 and now reads 0.3x
(useful as a "damaged pad" example, not for training).

The physics of the signal: pressing a pad **lowers** its ADC value. Unloaded
pads sit at about 2100-2300 (12-bit); a 200 g weight brings one down by
1200-1400. The analysis works in "8-bit-equivalent counts" = raw / 16 so that
numbers are comparable with the older 16x16 board; one such count is 12.9 mV.

## 2. The ready-made table

Run once (or ask for the files):

```powershell
.venv-calib\Scripts\python.exe sensor-data-collection\test\make_ml_dataset.py C:\tactile_data\weight_cal
```

This writes `ml_dataset.csv` (25 833 rows, one per frame, the 4 sessions) and
`ml_dataset.npz`. Columns:

| column | meaning |
|---|---|
| `session`, `pad`, `pad_row`, `pad_col` | which recording, which pad the weight was centred on (label for "where") |
| `t_s`, `frame_index` | time in the session, about 10 frames per second |
| `mass_g`, `load_N` | grams on the pad in this frame, 0 when empty (label for "how much") |
| `step`, `rep`, `order` | placement number, repetition (1-3), order of that repetition (ascending / descending / shuffled) |
| `phase` | `idle`, `loading` (weight just placed, response still creeping), `hold` (last 5 s of the placement, the plateau), `unloading` |
| `loaded` | 1 while a weight is on |
| `d_r1c1 ... d_r4c4` | **the features**: 16 cell responses, baseline-subtracted, pressure positive, 8-bit-equivalent counts |
| `raw_r1c1 ... raw_r4c4` | the raw 12-bit values, if you prefer to do your own preprocessing |

The on/off edges of every placement come from the loaded pad's own signal, not
from the key presses (those are a few seconds off), so the labels sit on the
data. `ml_dataset.npz` has the same thing as arrays: `X` (N, 4, 4) float32,
`y_mass` (N,), `y_pad` (N,) strings, `phase`, `session`.

Counts in the current set: 13 602 loaded frames, 3 120 plateau frames, five
masses (10 / 20 / 50 / 100 / 200 g), four pads, three repetitions each.

## 3. How to split: never random frames

Frames 0.1 s apart are near-duplicates. A random train/test split will give
you ~0 error and tell you nothing. Use one of these:

- **LORO, leave one repetition out**: train on repetitions 1-2 of every pad,
  test on repetition 3, rotate. Answers "how well does it do on a pad it has
  seen, on a new placement". This is the main number.
- **LOPO, leave one pad out**: train on three pads, test on the fourth.
  Answers "does it generalise to a pad it has never seen". Pads differ in gain
  (0.74-1.09) and in where they saturate, so this is much harder.
- Later, when the same pad is recorded on another day: **leave one session
  out**, which adds drift.

`ml_baseline.py` implements the first two; copy its `rep_of()` helper.

## 4. Numbers to beat

All on the four healthy pads, grams, 2026-10-31:

| model | frames | LORO MAE | LOPO MAE |
|---|---|---|---|
| physics: shared Hill curve + one gain per pad (`weight_live.py`) | plateau | 22 | 23 (gain from one 100 g weight) |
| sklearn HistGradientBoosting on the 16 responses (`ml_baseline.py`) | plateau | 27 | 58 |
| sklearn MLP 64-64 on the 16 responses (`ml_baseline.py`) | plateau | 11 | 473 (useless) |

Which pad: the argmax of the 16 responses is already 100 % right; a classifier
adds nothing. The interesting target is the grams.

Read the MLP row carefully. 11 g on seen pads beats the physics model because
the 16-cell *pattern* identifies the weight: the 100 g and 200 g pieces have
wider bases and spill more into the neighbours, so the net learns the
footprint of each piece, not only the force. That is real information for
"which of our five weights is it", and no information at all for an unknown
object; and the same net is hopeless on a pad it has not seen (485 g),
because nothing in it knows that pads differ by a gain. A frame-level model
therefore matches or beats the physics model on seen pads and is much worse
on unseen ones. Things that would make a learned model worth it:

- predicting through the creep: the plateau is reached 5-10 s after
  placement; a model that uses the time since contact (`phase`, `t_s` within
  a step) could report the right grams earlier;
- normalising per pad first (divide the responses by the pad's gain from
  `force_model.json`) and learning only the residual, so LOPO stops being a
  gain-estimation problem;
- more pads and more days: 16 pads x several days is the plan, the files will
  appear under the same root with the same layout.

## 5. Known traps

- **Saturation.** Response flattens around 1 N; on R1C1 100 g and 200 g read
  the same. No model can separate them on that pad.
- **Placement variance** dominates: the same weight re-placed on the same pad
  can differ by 10-25 % in counts. Expect ~20 g floor until a placement
  template is used.
- **R2C2 is damaged** (0.3x gain after a 20 N overload). Its session is in
  `weight_cal_old`; if you pull it in, treat it as its own domain.
- **Spill.** A 100 / 200 g weight's base is wider than a pad: 5-30 % of the
  response lands on a neighbour. The physics model sums the footprint; a
  per-cell model should see all 16 inputs, not only the peak.
- **Polarity and scale.** If you read `sensor_raw.csv` yourself: subtract the
  session's first-5-s median per cell, flip the sign (pressing lowers the
  ADC), divide by 16. `tactile_preprocess.load_raw/baseline/clean` do this.

## 6. Minimal example

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

d = np.load(r"C:\tactile_data\weight_cal\ml_dataset.npz")
X, y, pad, phase = d["X"].reshape(len(d["X"]), -1), d["y_mass"], d["y_pad"].astype(str), d["phase"].astype(str)
keep = (phase == "hold") & (pad != "R2C2")
X, y, pad = X[keep], y[keep], pad[keep]

test = pad == "R1C4"                      # leave one pad out
m = HistGradientBoostingRegressor().fit(X[~test], y[~test])
print("MAE on R1C4:", np.abs(m.predict(X[test]) - y[test]).mean(), "g")
```

For PyTorch: the same `X` (16 floats, scale by ~1/100) and `y` (grams / 100)
into a two-layer MLP is enough to reproduce the GBM numbers; the data is far
too small for anything convolutional to help.

## 7. Tools that already exist

| script (in `sensor-data-collection/test/`) | does |
|---|---|
| `weight_cal.py --pads all` | records new sessions, pad by pad |
| `weight_report.py <session>` | per-pad report |
| `weight_summary.py <root>` | all pads: table, gain map, shared fit |
| `weight_models.py <root>` | compares global / per-pad / shared models |
| `weight_eval.py <root>` | per-pad prediction quality figure |
| `weight_live.py train / live / replay` | the deployed physics model and the live "where and how much" window |
| `make_ml_dataset.py <root>` | this dataset |
| `ml_baseline.py <npz>` | the sklearn baselines above |
