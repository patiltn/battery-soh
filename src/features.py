"""
Feature extraction for cell degradation.

The organising idea: a discharge curve is a *function*, Q(V), not a scalar.
Most of the information about what is degrading lives in how the shape of
that function changes with cycle number, not in the endpoint capacity.  So
everything here works on curves interpolated onto a common voltage grid,
which makes cycle-to-cycle differencing well defined.

Three families of feature:

1. dQ(V) between two cycles  -- Severson et al. (Nat. Energy 2019) showed the
   variance of dQ(V) between cycle 100 and cycle 10 predicts log cycle life
   with ~10% error, using only the first 100 cycles and before any capacity
   fade is visible.  That is the benchmark to beat, and the reason this
   representation is the right one.

2. dQ/dV incremental capacity -- peaks correspond to phase transitions in the
   electrode.  Peak AREA tracks lithium inventory; peak POSITION shifts with
   resistance rise; peak WIDTH tracks active material loss.  So IC analysis
   partially separates degradation MODES, which endpoint capacity cannot.

3. Resistance and thermal -- ohmic resistance from the current-step voltage
   drop, plus per-cycle temperature statistics.  Independent of the Q(V)
   shape, and physically interpretable.

All features are computed from EARLY cycles only (default 10-100), so that
prediction targets (cycle life, capacity at cycle N) are genuinely in the
future relative to the inputs.  Leakage prevention starts here, at the
feature layer, not in the train/test split.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

# Common voltage grid for LFP discharge. Descending: discharge runs high->low.
V_GRID = np.linspace(3.5, 2.0, 1000)


# --------------------------------------------------------------------------
# Curve construction
# --------------------------------------------------------------------------
def q_of_v(V: np.ndarray, Q: np.ndarray, v_grid: np.ndarray = V_GRID) -> np.ndarray:
    """Interpolate discharge capacity onto a fixed voltage grid -> Q(V).

    V must be monotonically decreasing over the discharge (it is, apart from
    noise); we enforce monotonicity by sorting, then interpolate.  Points
    outside the measured voltage span are NaN rather than extrapolated --
    extrapolating here silently invents capacity.
    """
    order = np.argsort(V)
    v_s, q_s = V[order], Q[order]
    keep = np.concatenate([[True], np.diff(v_s) > 1e-9])   # strictly increasing
    v_s, q_s = v_s[keep], q_s[keep]
    out = np.interp(v_grid, v_s, q_s, left=np.nan, right=np.nan)
    in_range = (v_grid >= v_s.min()) & (v_grid <= v_s.max())
    out[~in_range] = np.nan

    # Enforce physical monotonicity.  Measurement noise on V reorders nearby
    # samples, which after interpolation shows up as Q(V) briefly decreasing --
    # i.e. capacity flowing backwards.  Q is cumulative charge drawn, so it can
    # only increase as voltage falls; the running maximum removes the artefact
    # without touching the genuine shape.
    valid = ~np.isnan(out)
    if valid.any():
        seg = out[valid]
        out[valid] = np.maximum.accumulate(seg) if v_grid[0] > v_grid[-1] \
            else np.maximum.accumulate(seg[::-1])[::-1]
    return out


def delta_q(q_late: np.ndarray, q_early: np.ndarray) -> np.ndarray:
    """dQ(V) = Q_late(V) - Q_early(V), elementwise on the shared grid."""
    return q_late - q_early


def dqdv(V: np.ndarray, Q: np.ndarray, v_grid: np.ndarray = V_GRID,
         smooth: float = 8.0) -> np.ndarray:
    """Incremental capacity dQ/dV on the voltage grid.

    Differentiating measured data amplifies noise as 1/dV, so smoothing is not
    cosmetic here -- without it the peak finder locks onto noise.  Gaussian
    smoothing in the voltage domain is applied BEFORE differencing; `smooth`
    is the kernel width in grid points.
    """
    q = q_of_v(V, Q, v_grid)
    valid = ~np.isnan(q)
    if valid.sum() < 20:
        return np.full_like(v_grid, np.nan)
    q_s = q.copy()
    q_s[valid] = gaussian_filter1d(q[valid], smooth, mode="nearest")
    d = np.full_like(v_grid, np.nan)
    d[valid] = np.gradient(q_s[valid], v_grid[valid])
    return np.abs(d)   # |dQ/dV|; sign is a convention of grid direction


# --------------------------------------------------------------------------
# Feature families
# --------------------------------------------------------------------------
def delta_q_features(dq: np.ndarray, prefix: str = "dQ") -> dict[str, float]:
    """Summary statistics of the dQ(V) curve.

    The variance is the Severson feature.  The rest (min, mean, skew,
    kurtosis) describe how the difference is distributed across voltage,
    which carries information about WHERE in the electrode the loss occurs.
    Logs are taken because these span orders of magnitude across cells and
    the downstream models are linear in the features.
    """
    d = dq[~np.isnan(dq)]
    if d.size < 20:
        return {f"{prefix}_{k}": np.nan
                for k in ("log_var", "log_min", "log_mean", "skew", "kurt")}
    eps = 1e-12
    return {
        f"{prefix}_log_var": float(np.log10(np.var(d) + eps)),
        f"{prefix}_log_min": float(np.log10(np.abs(d.min()) + eps)),
        f"{prefix}_log_mean": float(np.log10(np.abs(d.mean()) + eps)),
        f"{prefix}_skew": float(skew(d)),
        f"{prefix}_kurt": float(kurtosis(d)),
    }


def ic_peak_features(ic: np.ndarray, v_grid: np.ndarray = V_GRID,
                     n_peaks: int = 2, prefix: str = "IC") -> dict[str, float]:
    """Position, height and width of the dominant incremental-capacity peaks.

    Physical reading of the outputs:
      height / area  -> amount of material participating (lithium inventory)
      position       -> polarisation, moves with resistance growth
      width          -> heterogeneity / active material loss
    """
    out: dict[str, float] = {}
    valid = ~np.isnan(ic)
    if valid.sum() < 50:
        for i in range(n_peaks):
            out |= {f"{prefix}{i}_v": np.nan, f"{prefix}{i}_h": np.nan,
                    f"{prefix}{i}_w": np.nan}
        return out
    y, x = ic[valid], v_grid[valid]
    peaks, props = find_peaks(y, prominence=0.05 * np.nanmax(y), width=3)
    if peaks.size:
        order = np.argsort(props["prominences"])[::-1][:n_peaks]
        peaks, widths = peaks[order], props["widths"][order]
    else:
        widths = np.array([])
    for i in range(n_peaks):
        if i < peaks.size:
            out[f"{prefix}{i}_v"] = float(x[peaks[i]])
            out[f"{prefix}{i}_h"] = float(y[peaks[i]])
            out[f"{prefix}{i}_w"] = float(widths[i] * abs(x[1] - x[0]))
        else:
            out |= {f"{prefix}{i}_v": np.nan, f"{prefix}{i}_h": np.nan,
                    f"{prefix}{i}_w": np.nan}
    # |integral|: the grid runs high voltage -> low, so trapezoid over it is
    # signed negative.  The physical quantity is total capacity, so take the
    # magnitude rather than inheriting the grid's direction convention.
    out[f"{prefix}_area"] = float(abs(np.trapezoid(y, x)))
    return out


def ohmic_resistance(t: np.ndarray, I: np.ndarray, V: np.ndarray,
                     window: int = 5) -> float:
    """Estimate R0 from the instantaneous voltage step at current onset.

    At the moment current is applied, the RC branch has not yet responded
    (tau ~ tens of seconds), so the immediate drop is purely ohmic:
    R0 = dV / dI.  Anything later mixes in polarisation and is not R0.
    """
    di = np.abs(np.diff(I))
    if di.size == 0:
        return np.nan
    k = int(np.argmax(di))
    if di[k] < 1e-6 or k + window >= V.size:
        return np.nan
    dv = V[k] - V[k + 1]
    return float(abs(dv / (I[k + 1] - I[k])))


def thermal_features(T: np.ndarray, t: np.ndarray,
                     prefix: str = "T") -> dict[str, float]:
    """Temperature statistics for one cycle.

    The time integral matters more than the max: degradation rate is roughly
    Arrhenius in temperature, so accumulated thermal exposure is the physically
    relevant quantity, not the peak.
    """
    if T.size < 5:
        return {f"{prefix}_max": np.nan, f"{prefix}_avg": np.nan,
                f"{prefix}_integral": np.nan}
    return {
        f"{prefix}_max": float(T.max()),
        f"{prefix}_avg": float(T.mean()),
        f"{prefix}_integral": float(np.trapezoid(T - T.min(), t)),
    }


# --------------------------------------------------------------------------
# Per-cell assembly
# --------------------------------------------------------------------------
def cell_features(cycles: dict, summary: dict,
                  early: int = 10, late: int = 100) -> dict[str, float]:
    """Build one feature row for a cell, using ONLY cycles <= `late`.

    This is the unit the model sees: features from the first ~100 cycles,
    predicting something about the cell's future.  Nothing after `late` is
    touched -- which is the whole point.
    """
    avail = np.array(sorted(cycles.keys()))
    avail = avail[avail <= late]
    if avail.size < 5:
        raise ValueError(f"need >=5 cycles at or below cycle {late}")

    c_early = int(avail[np.argmin(np.abs(avail - early))])
    c_late = int(avail[-1])

    q_e = q_of_v(cycles[c_early]["V"], cycles[c_early]["Qd"])
    q_l = q_of_v(cycles[c_late]["V"], cycles[c_late]["Qd"])

    feats: dict[str, float] = {}
    feats |= delta_q_features(delta_q(q_l, q_e))
    feats |= ic_peak_features(dqdv(cycles[c_late]["V"], cycles[c_late]["Qd"]))

    # incremental-capacity drift between the same two cycles
    ic_e = ic_peak_features(dqdv(cycles[c_early]["V"], cycles[c_early]["Qd"]),
                            prefix="ICe")
    for i in range(2):
        for k in ("v", "h", "w"):
            feats[f"dIC{i}_{k}"] = feats.get(f"IC{i}_{k}", np.nan) - ic_e.get(f"ICe{i}_{k}", np.nan)

    # resistance growth over the early window
    r_e = ohmic_resistance(cycles[c_early]["t"], cycles[c_early]["I"], cycles[c_early]["V"])
    r_l = ohmic_resistance(cycles[c_late]["t"], cycles[c_late]["I"], cycles[c_late]["V"])
    feats["R0_early"] = r_e
    feats["dR0"] = (r_l - r_e) if np.isfinite(r_e) and np.isfinite(r_l) else np.nan

    feats |= thermal_features(cycles[c_late]["T"], cycles[c_late]["t"])

    # capacity trend over the early window (slope, not level)
    mask = summary["cycle"] <= late
    cyc, qd = summary["cycle"][mask], summary["QD"][mask]
    if cyc.size >= 3:
        feats["Q_slope_early"] = float(np.polyfit(cyc, qd, 1)[0])
        feats["Q_at_late"] = float(qd[-1])
        feats["Q_ratio"] = float(qd[-1] / qd[0])
    else:
        feats |= {"Q_slope_early": np.nan, "Q_at_late": np.nan, "Q_ratio": np.nan}

    return feats


def fleet_features(fleet: dict, early: int = 10, late: int = 100):
    """Feature matrix + cycle-life target for a whole fleet.

    Returns (X, y, cell_ids) with X as a list of dicts -- deliberately not a
    DataFrame here so this module stays dependency-light; the modelling layer
    does the conversion.
    """
    X, y, ids = [], [], []
    for cid, (cycles, summary, life) in fleet.items():
        try:
            X.append(cell_features(cycles, summary, early, late))
        except ValueError:
            continue
        y.append(life)
        ids.append(cid)
    return X, np.array(y, dtype=float), ids
