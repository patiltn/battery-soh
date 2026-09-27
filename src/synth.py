"""
Physics-based synthetic cell generator.

Purpose: a controlled ground-truth source for developing and testing the
feature layer before the real Severson data is available, and a permanent
fixture for the unit tests.  Every cell here has a KNOWN degradation
trajectory, so a feature that is supposed to track fade can be checked
against the truth rather than against a hopeful plot.

Model
-----
Open-circuit voltage: an LFP-like OCV(z) curve -- two plateaus separated by
a shallow step, which is what makes dQ/dV analysis informative for this
chemistry (the peaks in dQ/dV sit at the plateau edges and move/shrink as
the cell ages).

Terminal voltage under load (first-order ECM):

    V(t) = OCV(z(t)) - I*R0 - V1(t),     tau1 * dV1/dt + V1 = I*R1

Degradation, two independent modes:
  * LLI  (lithium inventory loss)  -> capacity fade, sqrt-in-cycle growth
  * LAM  (active material loss)    -> plateau shortening, linear in cycle
  * R0 growth                      -> ohmic resistance rise, linear + noise

These are the standard first-order pictures; the point is not fidelity to a
real cell but that the *structure* (fade, resistance rise, dQ/dV peak
movement) is present and its magnitude is known.
"""

from __future__ import annotations

import numpy as np

# Nominal cell parameters, loosely A123 APR18650M1A (LFP, 1.1 Ah nominal)
Q_NOM = 1.1          # Ah
V_MIN, V_MAX = 2.0, 3.6
R0_INIT = 0.015      # ohm
R1_INIT = 0.008      # ohm
TAU1 = 30.0          # s
T_AMB = 30.0         # degC
THERMAL_R = 12.0     # K/W  (cell-to-ambient)
REST_S = 20.0        # s of open-circuit rest before the discharge step


def ocv(z: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """Open-circuit voltage as a function of state of charge z in [0, 1].

    LFP-like: flat plateaus with a step between them.  `lam` (active material
    loss, 0..1) shortens the upper plateau, which is what shifts the dQ/dV
    peak positions as the cell ages.
    """
    z = np.clip(z, 0.0, 1.0)
    # two smoothed steps -> two plateaus
    step_lo = 1.0 / (1.0 + np.exp(-(z - 0.05) / 0.02))
    knee = 0.55 - 0.15 * lam          # plateau boundary migrates with LAM
    step_hi = 1.0 / (1.0 + np.exp(-(z - knee) / 0.06))
    v = 2.60 + 0.62 * step_lo + 0.22 * step_hi + 0.14 * z
    return v


def capacity(cycle: int, lli_rate: float, knee_cycle: float) -> float:
    """Usable capacity (Ah) at a given cycle.

    sqrt(cycle) SEI-growth regime, then an accelerating 'knee' -- the
    two-regime shape seen in real LFP ageing.
    """
    fade_sei = lli_rate * np.sqrt(cycle)
    fade_knee = 0.35 * np.exp((cycle - knee_cycle) / (0.18 * knee_cycle))
    return Q_NOM * max(0.0, 1.0 - fade_sei - fade_knee)


def resistance(cycle: int, r_rate: float, rng: np.random.Generator) -> float:
    """Ohmic resistance growth (ohm)."""
    return R0_INIT * (1.0 + r_rate * cycle) * (1.0 + 0.01 * rng.standard_normal())


def simulate_discharge(
    cyc: int,
    lli_rate: float,
    lam_rate: float,
    r_rate: float,
    knee_cycle: float,
    c_rate: float = 4.0,
    dt: float = 1.0,
    rng: np.random.Generator | None = None,
):
    """Simulate one constant-current discharge, returning Severson-like arrays.

    Returns dict with t (s), I (A), V (V), T (degC), Qd (Ah).
    """
    rng = rng or np.random.default_rng(0)
    q_max = capacity(cyc, lli_rate, knee_cycle)
    if q_max <= 0.05 * Q_NOM:
        return None
    lam = min(1.0, lam_rate * cyc)
    r0 = resistance(cyc, r_rate, rng)
    r1 = R1_INIT * (1.0 + 0.6 * r_rate * cyc)

    i_dis = c_rate * Q_NOM                      # A, constant current
    z, v1, temp = 1.0, 0.0, T_AMB
    t, I, V, T, Q = [], [], [], [], []
    q_drawn, clock = 0.0, 0.0

    # Rest at open circuit before the current step.  Without this the discharge
    # has no current transition and R0 = dV/dI is not identifiable -- the same
    # reason real test protocols include a rest before each pulse.
    for _ in range(int(REST_S / dt)):
        t.append(clock); I.append(0.0)
        V.append(float(ocv(np.array(z), lam)) + 0.0015 * rng.standard_normal())
        T.append(T_AMB + 0.05 * rng.standard_normal()); Q.append(0.0)
        clock += dt

    while z > 0.0:
        v_oc = float(ocv(np.array(z), lam))
        v1 += dt / TAU1 * (i_dis * r1 - v1)     # RC branch relaxation
        v_term = v_oc - i_dis * r0 - v1
        # lumped thermal: ohmic + polarisation heating, Newtonian cooling
        p_loss = i_dis ** 2 * r0 + i_dis * v1
        temp += dt * (p_loss * THERMAL_R - (temp - T_AMB)) / 90.0
        if v_term <= V_MIN:
            break
        t.append(clock)
        I.append(-i_dis)
        V.append(v_term + 0.0015 * rng.standard_normal())
        T.append(temp + 0.05 * rng.standard_normal())
        Q.append(q_drawn)
        dq = i_dis * dt / 3600.0                # Ah
        q_drawn += dq
        z -= dq / q_max
        clock += dt

    if len(t) < 20:
        return None
    return {
        "t": np.array(t),
        "I": np.array(I),
        "V": np.array(V),
        "T": np.array(T),
        "Qd": np.array(Q),
        "truth": {"capacity": q_max, "R0": r0, "lam": lam},
    }


def simulate_cell(
    n_cycles: int = 1200,
    lli_rate: float = 0.0025,
    lam_rate: float = 2.5e-4,
    r_rate: float = 2.0e-4,
    knee_cycle: float = 900.0,
    seed: int = 0,
    every: int = 1,
):
    """Simulate a full cell life. Returns (cycles dict, summary DataFrame-like dict)."""
    rng = np.random.default_rng(seed)
    cycles, summary = {}, {"cycle": [], "QD": [], "R0_true": [], "Tmax": [], "Tavg": []}
    for c in range(1, n_cycles + 1, every):
        out = simulate_discharge(c, lli_rate, lam_rate, r_rate, knee_cycle, rng=rng)
        if out is None:
            break
        cycles[c] = out
        summary["cycle"].append(c)
        summary["QD"].append(out["Qd"][-1])
        summary["R0_true"].append(out["truth"]["R0"])
        summary["Tmax"].append(out["T"].max())
        summary["Tavg"].append(out["T"].mean())
    summary = {k: np.array(v) for k, v in summary.items()}
    # cycle life: first cycle below 80% of nominal (EOL convention)
    below = np.where(summary["QD"] < 0.8 * Q_NOM)[0]
    cycle_life = int(summary["cycle"][below[0]]) if below.size else int(summary["cycle"][-1])
    return cycles, summary, cycle_life


def simulate_fleet(n_cells: int = 24, seed: int = 0, every: int = 5,
                   knee_noise: float = 0.15):
    """A fleet of cells with spread in degradation rates -> spread in cycle life.

    The knee cycle is COUPLED to the early degradation rates rather than drawn
    independently.  That coupling is the physical assumption the whole method
    rests on: cells whose early-cycle Q(V) is already drifting are the cells
    that fail sooner, because the same mechanisms (SEI growth, lithium plating)
    drive both.  If the knee were independent of the early rates, no feature of
    the first 100 cycles could predict cycle life, and a synthetic fixture
    built that way would wrongly make a working feature look useless.
    `knee_noise` sets how much cell-to-cell scatter breaks the coupling.
    """
    rng = np.random.default_rng(seed)
    fleet = {}
    for i in range(n_cells):
        lli = float(rng.uniform(0.0015, 0.0045))
        lam = float(rng.uniform(1e-4, 5e-4))
        r_r = float(rng.uniform(1e-4, 4e-4))
        # faster early degradation -> earlier knee, with scatter
        severity = (lli / 0.003) * (1.0 + 0.3 * r_r / 2.5e-4)
        knee = 1000.0 / severity * float(rng.lognormal(0.0, knee_noise))
        fleet[f"synth{i:03d}"] = simulate_cell(
            lli_rate=lli, lam_rate=lam, r_rate=r_r,
            knee_cycle=float(np.clip(knee, 300, 2000)),
            seed=i, every=every,
        )
    return fleet
