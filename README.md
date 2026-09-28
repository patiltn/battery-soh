# Battery State-of-Health: physics-informed early prediction of cycle life

Predicting lithium-ion cell degradation from the first 100 cycles, with a
feature layer built around the physics of what is actually degrading rather
than a sequence model on raw traces.

Dataset: [Severson et al., *Nature Energy* (2019)](https://data.matr.io/1/) —
124 LFP cells cycled to failure under 72 fast-charging policies.

## Results

From `python src/model.py` (elastic net on log₁₀ cycle life, the authors' own
train / primary-test / secondary-test split):

| split | n | mean err | median err |
|---|---|---|---|
| train | 42 | 5.5% | 4.3% |
| primary test | 42 | 26.7% | 5.4% |
| **secondary test (batch 3)** | **40** | **8.9%** | **6.4%** |

The secondary test set is the number that means something: forty cells from a
later batch under different charging policies, never seen in training or
model selection. 8.9% mean error there, from the first 100 cycles, before any
meaningful capacity fade is visible.

The primary test set's mean (26.7%) and median (5.4%) differ by a factor of
five. That gap is **one cell**, and it turned out to be the most interesting
thing in the project.

## The b2c1 problem

Cell b2c1 dies at 148 cycles. The 20-feature model predicts 2056.

The obvious explanation — an abrupt internal fault that early cycles could
not have predicted — is wrong. By cycle 100 b2c1 was the most visibly
degraded cell in the dataset: dQ(V) variance ~10× the cohort median, 9%
capacity already lost where the median cell had lost none. The information
was there. The model ignored it.

Fitting the same split with different feature counts:

| features | primary mean | primary median | b2c1 prediction (true 148) |
|---|---|---|---|
| `dQ_log_var` only | 14.5% | 12.4% | 269 |
| paper's 3 features | 13.7% | 12.1% | **208** |
| all 20 | 37.9% | **5.4%** | **2056** |

Two mechanisms, both visible in the fitted model:

**Collinearity.** `Q_ratio`, `Q_at_late` and `Q_slope_early` are near
duplicates. The fit puts a *negative* coefficient on `Q_ratio` despite its
*positive* (+0.32) correlation with cycle life. Those opposing terms cancel
for typical cells and diverge for extreme ones.

**Extrapolation.** b2c1's `dQ_log_var` is −2.73; the training range is −5.01
to −2.73. It sits exactly on the boundary, which is where a model with many
correlated features is least constrained by data.

### The flag that catches it

`src/results.py` flags any test cell whose features fall outside the
per-feature training range. With the 3-feature model it flags **2 of 82**
test cells, and b2c1 is one of them — actionable: that prediction can be
withheld or widened rather than reported. With all 20 features it flags **48
of 82** and the median error inside and outside barely differs (5.3% vs
7.2%), so the flag carries no information.

A cell is flagged if *any* feature is out of range, so in high dimensions
almost everything is flagged. That is not a defect of the check; it is what
extrapolation looks like when you have 20 correlated features and 124 cells.
The model is extrapolating most of the time — it simply does not say so.

## The other failure: the long-lived cells

b2c1 is one direction of error. The other is systematic. Every cell above
1600 cycles is **under**-predicted:

| cell | true | predicted | error |
|---|---|---|---|
| b1c2 | 2237 | 1243 | 44% |
| b3c38 | 1935 | 1112 | 42% |
| b3c7 | 1836 | 1316 | 28% |
| b3c45 | 1801 | 1427 | 21% |
| b3c16 | 1638 | 1303 | 20% |

This is regression toward the mean at the top of the range, and it is what a
regularised linear model is *supposed* to do when the signal runs out: shrink
toward the centre. The features distinguish a 500-cycle cell from a
1000-cycle one; they do not distinguish 1600 from 2200, because by cycle 100
those cells have barely begun to degrade and look alike.

Operationally this matters less than b2c1 — being told a cell will last 1100
cycles when it lasts 1900 is a conservative error — but it bounds the useful
range of the method from above as well as below.

## Trade-off

So more features bought a better median (5.4% vs 12.1%) and lost the one cell
whose failure mattered most. For a manufacturer those are not comparable:
being 7 points better on healthy cells does not pay for predicting 2000
cycles of remaining life on a cell that has 48. `src/results.py` reports the
feature sets side by side rather than picking a winner, and flags predictions
that fall outside the training range instead of returning them with unearned
confidence.

## Leakage

Leakage prevention starts in the feature layer, not the train/test split.
`test_cell_features_ignore_cycles_after_cutoff` deletes every cycle after the
cutoff and asserts that not one feature value changes — a split-level check
would pass even if the features were peeking ahead.

Splits are across **cells**, never within a cell. Each cell reduces to one
row, so a random row split cannot accidentally put cycle 40 and cycle 60 of
the same cell on opposite sides.

`compare_leaky_vs_honest` in `src/model.py` selects features once on all
cells and once inside each fold. On this dataset the gap is **0.0 percentage
points** — the ΔQ(V) features are chosen so consistently that seeing the test
cells does not help. A negative result, reported rather than quietly dropped.

## The feature layer

A discharge curve is a function `Q(V)`, not a scalar, and most of the
information about *what* is degrading lives in how that function's shape
changes with cycle number.

| family | physical reading |
|---|---|
| `ΔQ(V)` statistics | variance between cycles 100 and 10 predicts log cycle life before capacity fade is visible (Severson et al.); here `corr = −0.93` |
| `dQ/dV` incremental capacity | peak **area** → lithium inventory; **position** → polarisation; **width** → active material loss. Partially separates degradation *modes* |
| resistance & thermal | `R0` from the current-step voltage drop, before the RC branch responds; thermal exposure as a time integral (degradation is ~Arrhenius in T) |

Two things the data forced:

`q_of_v` masks to the **discharge branch** using the measured current. A raw
cycle record contains charge and discharge, which sit at different voltages
for the same capacity (hysteresis plus an IR drop that reverses sign with
current). Interpolating over both blends two distinct curves. Masking cut
disagreement with the published `Qdlin` from 0.057 Ah to 0.035 Ah.

The **second incremental-capacity peak does not exist** in this data. At the
4C discharge rate used here, polarisation smears the LFP plateaux together
and only one peak resolves — so six features are NaN for all 124 cells and
are dropped rather than imputed. A feature that is absent is not a feature
that is missing.

## Validation

Tests check physical behaviour, not just execution: `Q(V)` monotone, `dQ/dV`
finds the plateau peaks, integrated `dQ/dV` falls with age, estimated `R0`
recovers the simulator's true `R0` within 20%.

More usefully, `tests/test_real_data.py` checks our `Q(V)` against the
**published `Qdlin`** — an independent implementation of the same quantity by
the original authors. Median agreement 0.002 Ah across ten cells. Tests
against our own simulator can only catch internal inconsistency; this one
catches being wrong.

Those tests skip automatically when the raw `.mat` files are absent, so the
suite runs on a clean clone without the 8 GB download.

## Data cleaning

Reproduces the authors' own `Load Data.ipynb` — these are not optional:

- **5 batch-1 cells** never reach 80% capacity → excluded
- **6 batch-3 cells** sit on noisy channels → excluded
- **5 batch-1 cells continued testing in batch 2** under new keys → cycle
  lives summed, duplicates removed. Skip this and five cells carry targets
  wrong by 200–1000 cycles, with nothing downstream to flag it.
- **batch 1 records a zero** as its first summary capacity (batches 2 and 3 do
  not) → dropped before fitting. Left in, it makes `Q_ratio` infinite and
  drags the slope fit through an artefact. Since batch membership correlates
  with charging policy and hence cycle life, that lets a model learn *which
  batch* a cell came from instead of how it ages.

46 + 48 + 46 = 140 cells, minus 16 → **124**, matching the paper exactly.

## Synthetic fixture

`src/synth.py` simulates cells from a first-order ECM (LFP-like OCV, RC
branch, lumped thermal model) with three degradation modes. It gives the
tests known ground truth.

**Limitation, stated plainly:** on the synthetic fleet, early capacity slope
is the strongest predictor (r ≈ 0.94). On the real data it is weak (+0.39)
and `ΔQ(V)` variance dominates (−0.93) — which is exactly the paper's point
and the opposite of the fixture. The simulator ties capacity fade too
directly to the underlying rates. It is a correctness check on the code,
never evidence that the features work.

## Layout

```
src/features.py   Q(V) interpolation, ΔQ(V), dQ/dV, resistance, thermal
src/loader.py     raw MATLAB v7.3 batch files → cleaned cells → feature cache
src/model.py      elastic net, paper split, cell-wise CV, leakage check
src/results.py    feature-set comparison, extrapolation flagging
src/synth.py      physics-based cell simulator (test fixture)
tests/            17 tests; 6 validate against the published dataset
```

## Run

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # 17 pass; 6 skip without raw data

# optional: download the three batch .mat files to data/raw/ from data.matr.io
python src/loader.py                # → data/features.npz (20 KB, from 8 GB)
python src/model.py
python src/results.py
```

The feature cache is the deliverable: once written, the raw 8 GB can be
deleted and everything downstream runs from 20 KB.

## Status

- [x] Feature extraction + physics layer
- [x] Test suite against known ground truth
- [x] Loader for the real Severson `.mat` files
- [x] Validation against the published `Qdlin`
- [x] Baselines: elastic net, paper's split, cell-wise CV
- [x] Leakage check; feature-set comparison; extrapolation flagging
- [ ] State-of-health task (capacity at cycle *n*), where a cycle-wise vs
      cell-wise split demonstrates leakage far more dramatically
- [ ] Per-cell prediction intervals rather than point estimates
- [ ] Separate treatment of the >1600-cycle regime, where the early-cycle
      signal saturates and predictions shrink toward the mean
