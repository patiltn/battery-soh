"""
Feature-set comparison and extrapolation flagging.

This module exists because of one cell.  b2c1 dies at 148 cycles, and by
cycle 100 it is already the most visibly degraded cell in the dataset: its
dQ(V) variance is ~10x the cohort median and it has lost 9% capacity while
the median cell has lost none.  The 20-feature model predicts it will last
2056 cycles.  The 3-feature model predicts 208.

That is the whole argument for what follows.  More features gave a better
median error (5.4% vs 12.1% on the primary test set) and a catastrophic
error on the one cell whose failure mattered most.  For a manufacturer those
are not comparable quantities: being 7 points better on healthy cells does
not pay for telling you that a failing cell has 2000 cycles left.

Two mechanisms are at work, and both are visible in the data:

* Collinearity.  Q_ratio, Q_at_late and Q_slope_early are near-duplicates of
  each other.  The fit uses large opposing coefficients on them -- Q_ratio
  gets a NEGATIVE weight despite correlating POSITIVELY (+0.32) with cycle
  life.  Those cancelling terms behave for typical cells and diverge for
  extreme ones.
* Extrapolation.  b2c1's dQ_log_var sits exactly at the edge of the training
  range (-2.73, against a training span of -5.01 to -2.73).  A model with
  many correlated features is least constrained precisely there.

So the module reports feature sets side by side rather than selecting one,
and flags predictions that fall outside the training range instead of
returning them with unearned confidence.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import load_cache, make_model, paper_split

# Ordered by increasing capacity.  "paper3" is the published discharge model's
# shape: the dQ(V) statistics plus an endpoint capacity.
FEATURE_SETS = {
    "variance_only": ["dQ_log_var"],
    "paper3": ["dQ_log_var", "dQ_log_min", "Q_at_late"],
    "all": None,            # every column in the cache
}


def in_training_range(X_train: np.ndarray, X_test: np.ndarray,
                      tol: float = 0.0) -> np.ndarray:
    """Boolean mask: which test rows lie inside the per-feature training range.

    Deliberately crude -- a per-feature min/max box, not a density estimate.
    The point is not to model the training distribution but to answer a
    question the model cannot: "have I seen anything like this before?"  A
    prediction outside the box is an extrapolation, and a linear model
    extrapolates along whatever direction its collinear terms happen to
    define, which is unconstrained by data.

    `tol` widens the box by a fraction of each feature's training range, for
    when a hard boundary is too brittle.

    The box does not survive many dimensions.  A cell is flagged if ANY single
    feature is out of range, so with 20+ features nearly every cell is flagged
    and the signal is gone -- on the synthetic fixture, 33 of 40 test cells.
    That is not a defect of the check so much as a restatement of the problem:
    in high dimensions almost every point is near a boundary, and a model with
    that many correlated features is extrapolating far more often than its
    confident-looking outputs suggest. Use the flag with the small feature
    sets, where "have I seen anything like this?" still has a useful answer.
    """
    lo, hi = X_train.min(axis=0), X_train.max(axis=0)
    if tol:
        pad = (hi - lo) * tol
        lo, hi = lo - pad, hi + pad
    return ((X_test >= lo) & (X_test <= hi)).all(axis=1)


def _fit_predict(X, y, tr, idx, cols, regularised: bool):
    model = (make_model() if regularised
             else make_pipeline(StandardScaler(), LinearRegression()))
    model.fit(X[np.ix_(tr, cols)], np.log10(y[tr]))
    return 10.0 ** model.predict(X[np.ix_(idx, cols)]), model


def compare_feature_sets(X, y, ids, names, regularised: bool = False,
                         verbose: bool = True):
    """Evaluate each feature set on the paper's splits.

    Reports median alongside mean, and the worst single cell.  A mean alone
    hides exactly the failure this module is about: on the primary test set
    the 20-feature model's mean (26.7%) and median (5.4%) differ by a factor
    of five, and the entire gap is one cell.
    """
    tr, te, sec = paper_split(ids)
    rows = []

    for label, cols_names in FEATURE_SETS.items():
        cols = (list(range(len(names))) if cols_names is None
                else [names.index(n) for n in cols_names if n in names])
        if not cols:
            continue
        for split_name, idx in (("primary", te), ("secondary", sec)):
            pred, _ = _fit_predict(X, y, tr, idx, cols, regularised)
            err = np.abs(pred - y[idx]) / y[idx] * 100.0
            worst = int(np.argmax(err))
            rows.append({
                "features": label, "n_feat": len(cols), "split": split_name,
                "mean": float(err.mean()), "median": float(np.median(err)),
                "worst_cell": ids[idx[worst]], "worst_err": float(err[worst]),
                "n_extrapolated": int((~in_training_range(X[np.ix_(tr, cols)],
                                                          X[np.ix_(idx, cols)])).sum()),
            })

    if verbose:
        print(f"{'features':14s} {'n':>3s} {'split':10s} {'mean %':>8s} "
              f"{'median %':>9s} {'worst cell':>11s} {'worst %':>9s} {'extrap':>7s}")
        for r in rows:
            print(f"{r['features']:14s} {r['n_feat']:3d} {r['split']:10s} "
                  f"{r['mean']:8.1f} {r['median']:9.1f} {r['worst_cell']:>11s} "
                  f"{r['worst_err']:9.1f} {r['n_extrapolated']:7d}")
    return rows


def flag_extrapolation(X, y, ids, names, cols_names=None, verbose: bool = True):
    """Which test cells are outside the training box, and how wrong are they?

    If extrapolated cells are systematically worse, the flag is worth acting
    on: a prediction can be withheld or widened rather than reported as if it
    were as trustworthy as the rest.
    """
    tr, te, sec = paper_split(ids)
    cols = (list(range(len(names))) if cols_names is None
            else [names.index(n) for n in cols_names])
    idx = np.concatenate([te, sec])
    pred, _ = _fit_predict(X, y, tr, idx, cols, regularised=False)
    err = np.abs(pred - y[idx]) / y[idx] * 100.0
    inside = in_training_range(X[np.ix_(tr, cols)], X[np.ix_(idx, cols)])

    if verbose:
        print(f"inside training range : n={inside.sum():3d}  "
              f"mean {err[inside].mean():5.1f}%  median {np.median(err[inside]):5.1f}%")
        if (~inside).any():
            print(f"outside (extrapolated): n={(~inside).sum():3d}  "
                  f"mean {err[~inside].mean():5.1f}%  median {np.median(err[~inside]):5.1f}%")
            for i in np.where(~inside)[0]:
                print(f"    {ids[idx[i]]:8s} true {y[idx[i]]:6.0f}  "
                      f"pred {pred[i]:6.0f}  err {err[i]:6.1f}%")
        else:
            print("outside (extrapolated): none")
    return {"inside": inside, "err": err, "ids": [ids[i] for i in idx]}


def main(cache: str = "data/features.npz"):
    X, y, ids, names = load_cache(cache)
    print(f"{X.shape[0]} cells, {X.shape[1]} features\n")

    print("=== feature sets, unregularised (isolates the collinearity effect) ===")
    compare_feature_sets(X, y, ids, names, regularised=False)

    print("\n=== feature sets, elastic net ===")
    compare_feature_sets(X, y, ids, names, regularised=True)

    print("\n=== extrapolation check, all features ===")
    flag_extrapolation(X, y, ids, names)

    print("\n=== extrapolation check, paper3 ===")
    flag_extrapolation(X, y, ids, names,
                       [n for n in FEATURE_SETS["paper3"] if n in names])


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
