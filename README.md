# Battery State-of-Health: physics-informed feature extraction

Predicting lithium-ion cell degradation from early-cycle data, with the
feature layer built around the physics of what is actually degrading rather
than a generic sequence model on raw traces.

Target dataset: [Severson et al., *Nature Energy* (2019)](https://data.matr.io/1/)
— 124 LFP cells cycled to failure under 72 fast-charging policies.

## The idea

A discharge curve is a function `Q(V)`, not a scalar. Most of the information
about *what* is degrading lives in how the shape of that function changes with
cycle number, not in the endpoint capacity. Everything here works on curves
interpolated onto a common voltage grid, which makes cycle-to-cycle
differencing well defined.

Three feature families:

| Family | Physical reading |
|---|---|
| `ΔQ(V)` statistics | Variance of `ΔQ` between cycles 100 and 10 predicts log cycle life to ~10% (Severson et al.), *before any capacity fade is visible* |
| `dQ/dV` incremental capacity | Peak **area** → lithium inventory; **position** → polarisation/resistance; **width** → active material loss. Partially separates degradation *modes*, which capacity alone cannot |
| Resistance & thermal | `R0` from the current-step voltage drop, before the RC branch responds; thermal exposure as a time integral (degradation is ~Arrhenius in T) |

All features are computed from cycles ≤ 100 only, so prediction targets are
genuinely in the future relative to the inputs.

## Leakage

Leakage prevention starts at the feature layer, not the train/test split.
`test_cell_features_ignore_cycles_after_cutoff` deletes every cycle after the
cutoff and asserts that not a single feature value changes — a split-level
check would pass even if the features were peeking ahead.

When modelling is added, splits are **across cells, never within a cell**.
Splitting cycles randomly leaks a cell's own future into its training data and
produces error figures that do not survive deployment.

## Synthetic fixture

`src/synth.py` simulates cells from a first-order ECM (LFP-like OCV, RC
branch, lumped thermal model) with three degradation modes: lithium inventory
loss, active material loss, and resistance growth. It exists so the tests have
known ground truth.

One design note. Drawing each cell's failure knee *independently* of its early
degradation rates made the `ΔQ(V)` feature show almost no correlation with
cycle life — correctly, because in that world the early cycles contain no
information about the future. Coupling the knee to the early rates is the
physical assumption the entire method rests on, and making it explicit in the
simulator is the point.

**Limitation:** on the synthetic fleet, early capacity slope is the strongest
predictor (r ≈ 0.94). On real Severson data it is famously *not* predictive,
which is precisely why `ΔQ(V)` was interesting. The simulator ties capacity
fade too directly to the underlying rates. Treat it as a correctness check on
the code, never as evidence the features work.

## Layout

```
src/features.py   Q(V) interpolation, ΔQ(V), dQ/dV, resistance, thermal
src/synth.py      physics-based cell simulator (test fixture)
tests/            11 tests, checking physical behaviour not just execution
```

## Run

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Status

- [x] Feature extraction + physics layer
- [x] Test suite against known ground truth
- [ ] Loader for the real Severson `.mat` files (MATLAB v7.3 → `h5py`)
- [ ] Baselines: regularised linear model on `ΔQ(V)` features (the paper's benchmark)
- [ ] Cell-wise cross-validation, leaky-vs-honest split comparison
