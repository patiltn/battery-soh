"""
Tests for the feature layer.

The tests check that features respond CORRECTLY to known changes in the
underlying cell, not merely that the code runs.  Each one states the physical
expectation it encodes, so a failure tells you which piece of physics the
implementation has stopped respecting.

Run:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import (  # noqa: E402
    V_GRID, cell_features, delta_q, delta_q_features, dqdv,
    ic_peak_features, ohmic_resistance, q_of_v,
)
from synth import R0_INIT, simulate_cell, simulate_discharge  # noqa: E402


@pytest.fixture(scope="module")
def cell():
    return simulate_cell(n_cycles=600, seed=3, every=10)


# --------------------------------------------------------------------------
# Curve construction
# --------------------------------------------------------------------------
def test_q_of_v_monotone_and_bounded(cell):
    """Q(V) must be non-decreasing as voltage falls: capacity only accumulates."""
    cycles, _, _ = cell
    c = sorted(cycles)[5]
    q = q_of_v(cycles[c]["V"], cycles[c]["Qd"])
    valid = q[~np.isnan(q)]
    assert valid.size > 100
    assert np.all(np.diff(valid) >= -1e-6)


def test_q_of_v_no_extrapolation(cell):
    """Outside the measured voltage span the curve is NaN, never invented."""
    cycles, _, _ = cell
    c = sorted(cycles)[5]
    q = q_of_v(cycles[c]["V"], cycles[c]["Qd"])
    below = V_GRID < cycles[c]["V"].min()
    assert np.all(np.isnan(q[below]))


# --------------------------------------------------------------------------
# dQ(V)
# --------------------------------------------------------------------------
def test_delta_q_zero_for_identical_cycles(cell):
    """A cycle differenced against itself gives exactly zero."""
    cycles, _, _ = cell
    c = sorted(cycles)[3]
    q = q_of_v(cycles[c]["V"], cycles[c]["Qd"])
    d = delta_q(q, q)
    assert np.nanmax(np.abs(d)) == 0.0


def test_delta_q_variance_grows_with_separation(cell):
    """Cycles further apart in life must differ more: var dQ(V) increases."""
    cycles, _, _ = cell
    cs = sorted(cycles)
    q0 = q_of_v(cycles[cs[1]]["V"], cycles[cs[1]]["Qd"])
    variances = []
    for idx in (5, 15, 30):
        q = q_of_v(cycles[cs[idx]]["V"], cycles[cs[idx]]["Qd"])
        variances.append(delta_q_features(delta_q(q, q0))["dQ_log_var"])
    assert variances[0] < variances[1] < variances[2]


# --------------------------------------------------------------------------
# Incremental capacity
# --------------------------------------------------------------------------
def test_dqdv_finds_plateau_peaks(cell):
    """LFP has plateaux; |dQ/dV| must show peaks at their edges."""
    cycles, _, _ = cell
    c = sorted(cycles)[3]
    ic = dqdv(cycles[c]["V"], cycles[c]["Qd"])
    feats = ic_peak_features(ic)
    assert np.isfinite(feats["IC0_v"]) and np.isfinite(feats["IC0_h"])
    assert 2.0 <= feats["IC0_v"] <= 3.5


def test_ic_area_falls_with_age(cell):
    """Integrated dQ/dV is total capacity, so it must shrink as the cell ages."""
    cycles, _, _ = cell
    cs = sorted(cycles)
    early = ic_peak_features(dqdv(cycles[cs[2]]["V"], cycles[cs[2]]["Qd"]))["IC_area"]
    late = ic_peak_features(dqdv(cycles[cs[-3]]["V"], cycles[cs[-3]]["Qd"]))["IC_area"]
    assert late < early


def test_dqdv_smoothing_suppresses_noise():
    """Differentiation amplifies noise; heavier smoothing must reduce roughness."""
    out = simulate_discharge(10, 0.0025, 2.5e-4, 2e-4, 900.0)
    rough = dqdv(out["V"], out["Qd"], smooth=1.0)
    smooth = dqdv(out["V"], out["Qd"], smooth=20.0)
    r = np.nanstd(np.diff(rough[~np.isnan(rough)]))
    s = np.nanstd(np.diff(smooth[~np.isnan(smooth)]))
    assert s < r


# --------------------------------------------------------------------------
# Resistance
# --------------------------------------------------------------------------
def test_ohmic_resistance_recovers_true_r0():
    """R0 from the current-step drop must match the simulator's R0 within 20%.

    Tolerance is loose because the estimate uses one noisy voltage sample; the
    point is that it is unbiased, not that it is precise.
    """
    out = simulate_discharge(1, 0.0025, 2.5e-4, 2e-4, 900.0)
    r_est = ohmic_resistance(out["t"], out["I"], out["V"])
    assert np.isfinite(r_est)
    assert abs(r_est - out["truth"]["R0"]) / out["truth"]["R0"] < 0.20


def test_resistance_grows_with_cycle():
    """Ohmic resistance rises with age -- the sign of the trend must be right."""
    early = simulate_discharge(1, 0.0025, 2.5e-4, 5e-4, 900.0)
    late = simulate_discharge(500, 0.0025, 2.5e-4, 5e-4, 900.0)
    assert late["truth"]["R0"] > early["truth"]["R0"] > 0.9 * R0_INIT


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------
def test_cell_features_ignore_cycles_after_cutoff(cell):
    """Features must depend ONLY on cycles <= `late`.

    Deleting every later cycle must not change a single feature value.  This is
    the test that actually prevents look-ahead: a split-level check would pass
    even if the feature layer were peeking into the future.
    """
    cycles, summary, _ = cell
    full = cell_features(cycles, summary, early=10, late=100)
    truncated_cycles = {k: v for k, v in cycles.items() if k <= 100}
    mask = summary["cycle"] <= 100
    truncated_summary = {k: v[mask] for k, v in summary.items()}
    trunc = cell_features(truncated_cycles, truncated_summary, early=10, late=100)
    for k in full:
        a, b = full[k], trunc[k]
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b), f"{k} leaked"


def test_cell_features_reject_insufficient_history(cell):
    """Too few early cycles is an error, not a silently degraded feature row."""
    cycles, summary, _ = cell
    few = {k: v for k, v in cycles.items() if k <= 30}
    with pytest.raises(ValueError):
        cell_features(few, summary, early=10, late=100)
