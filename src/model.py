"""
Baseline models for early prediction of cycle life.

Target is log10(cycle life), as in Severson et al.  Two reasons, both worth
stating: cycle life spans 150-2300 cycles here, so absolute errors are
dominated by the long-lived cells, and the error that matters operationally
is relative ("within 10%") rather than absolute ("within 100 cycles").
Predictions are exponentiated back before any error is reported, so every
number below is a percentage error on actual cycle life.

Model is a regularised linear one (elastic net).  With 124 cells and ~20
features, a linear model with strong regularisation is not a placeholder for
something better -- it is the right complexity for the data.  Anything with
more capacity will fit the noise, and on a dataset this small the honest
error bars will not distinguish it from this.

Three evaluations, deliberately not one:

1. `evaluate_paper_split`  -- the authors' own train / primary-test /
   secondary-test partition, so the numbers are comparable to the published
   ones instead of to nothing.
2. `evaluate_cv`           -- k-fold across CELLS.  Uses the data better but
   is not comparable to the paper.
3. `compare_leaky_vs_honest` -- the same model evaluated two ways, one of
   which leaks.  See the docstring there; this is the point of the module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
SEED = 0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_cache(path: str | Path = "data/features.npz"):
    """Load the feature cache written by `loader.cache_features`."""
    d = np.load(Path(path), allow_pickle=True)
    X, y = d["X"], d["y"]
    ids = [str(s) for s in d["ids"]]
    names = [str(s) for s in d["feature_names"]]
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not finite.all():
        print(f"[load_cache] dropping {(~finite).sum()} cells with non-finite values")
    return X[finite], y[finite], [i for i, k in zip(ids, finite) if k], names


def _cell_order(ids: list[str]) -> np.ndarray:
    """Indices sorting cells by (batch, cell number).

    The cache is keyed alphabetically, so 'b1c10' sorts before 'b1c2'.  The
    paper's split is positional -- every other cell -- so reproducing it needs
    the cells in the same NUMERIC order the authors used, not lexicographic.
    """
    def key(k: str):
        b, c = k[1:].split("c")
        return int(b), int(c)
    return np.array(sorted(range(len(ids)), key=lambda i: key(ids[i])))


def paper_split(ids: list[str]):
    """Reproduce the authors' train / primary test / secondary test split.

    Batches 1 and 2 are interleaved into train and primary test (every other
    cell); batch 3 is held out entirely as the secondary test set.  Batch 3
    used different charging policies and was run later, so it is the closest
    thing here to "cells you have never seen from a build you have not
    characterised" -- the only number that speaks to deployment.
    """
    order = _cell_order(ids)
    ordered = [ids[i] for i in order]
    b12 = np.array([i for i, k in enumerate(ordered) if not k.startswith("b3")])
    b3 = np.array([i for i, k in enumerate(ordered) if k.startswith("b3")])
    train_local = b12[1::2]
    test_local = b12[0::2]
    return order[train_local], order[test_local], order[b3]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def make_model(seed: int = SEED):
    """Standardise, then elastic net with the penalty chosen by inner CV.

    Standardisation is inside the pipeline so it is refitted on each training
    fold; fitting the scaler on all the data before splitting is itself a
    (mild) leak, and putting it in the pipeline makes that structurally
    impossible rather than merely intended.
    """
    return make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=L1_RATIOS, cv=5, max_iter=50_000,
                     random_state=seed),
    )


def _errors(y_true: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    """Error metrics on actual cycle life, not on the log target."""
    pred = 10.0 ** log_pred
    pct = np.abs(pred - y_true) / y_true * 100.0
    return {
        "mean_pct_err": float(pct.mean()),
        "median_pct_err": float(np.median(pct)),
        "rmse_cycles": float(np.sqrt(((pred - y_true) ** 2).mean())),
        "n": int(y_true.size),
    }


# --------------------------------------------------------------------------
# Evaluations
# --------------------------------------------------------------------------
def evaluate_paper_split(X, y, ids, names=None, verbose: bool = True):
    """Fit on the paper's training cells, report both test sets."""
    tr, te, sec = paper_split(ids)
    model = make_model()
    model.fit(X[tr], np.log10(y[tr]))

    out = {
        "train": _errors(y[tr], model.predict(X[tr])),
        "primary_test": _errors(y[te], model.predict(X[te])),
        "secondary_test": _errors(y[sec], model.predict(X[sec])),
    }
    if verbose:
        print(f"{'split':16s} {'n':>4s} {'mean %':>8s} {'median %':>9s} {'RMSE':>8s}")
        for k, v in out.items():
            print(f"{k:16s} {v['n']:4d} {v['mean_pct_err']:8.1f} "
                  f"{v['median_pct_err']:9.1f} {v['rmse_cycles']:8.0f}")
        if names is not None:
            enet = model[-1]
            nz = [(n, c) for n, c in zip(names, enet.coef_) if abs(c) > 1e-8]
            nz.sort(key=lambda z: -abs(z[1]))
            print(f"\nselected {len(nz)}/{len(names)} features "
                  f"(alpha={enet.alpha_:.4g}, l1_ratio={enet.l1_ratio_})")
            for n, c in nz[:8]:
                print(f"  {n:16s} {c:+.4f}")
    return out


def evaluate_cv(X, y, n_splits: int = 5, seed: int = SEED, verbose: bool = True):
    """k-fold cross-validation ACROSS CELLS.

    Every row here is one cell, so a plain KFold already splits by cell.  That
    is not an accident of the data layout -- it is why the feature layer
    reduces each cell to a single row.  Had we kept one row per cycle, a
    random split would put cycle 40 of a cell in train and cycle 60 of the
    same cell in test, and the reported error would be meaningless.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.full(y.shape, np.nan)
    for tr, te in kf.split(X):
        m = make_model(seed)
        m.fit(X[tr], np.log10(y[tr]))
        preds[te] = m.predict(X[te])
    res = _errors(y, preds)
    if verbose:
        print(f"{n_splits}-fold CV across cells: mean {res['mean_pct_err']:.1f}% "
              f"| median {res['median_pct_err']:.1f}% | RMSE {res['rmse_cycles']:.0f} cycles")
    return res


def compare_leaky_vs_honest(X, y, names, k: int = 5, n_splits: int = 5,
                            seed: int = SEED, verbose: bool = True):
    """Feature selection inside the CV loop versus outside it.

    Both arms select the `k` features most correlated with the target and fit
    the same model with the same folds.  The only difference is WHEN the
    selection happens:

      leaky  -- choose the k features using ALL cells, then cross-validate.
                The held-out cells influenced which features were chosen, so
                they were never really held out.
      honest -- choose the k features using only each fold's training cells.

    The leaky number is the one that appears in a great many published
    pipelines and portfolio repos, because it is what you get by doing the
    obvious thing in the obvious order.  The gap between the two is the size
    of the lie, and it is reported here rather than avoided quietly.
    """
    ly = np.log10(y)

    def top_k(idx_rows, kk):
        cors = []
        for j in range(X.shape[1]):
            c = X[idx_rows, j]
            cors.append(abs(np.corrcoef(c, ly[idx_rows])[0, 1]) if c.std() > 0 else 0.0)
        return np.argsort(cors)[::-1][:kk]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_rows = np.arange(X.shape[0])

    leaky_cols = top_k(all_rows, k)          # <- selection sees every cell
    leaky_pred = np.full(y.shape, np.nan)
    honest_pred = np.full(y.shape, np.nan)

    for tr, te in kf.split(X):
        m = make_model(seed)
        m.fit(X[np.ix_(tr, leaky_cols)], ly[tr])
        leaky_pred[te] = m.predict(X[np.ix_(te, leaky_cols)])

        cols = top_k(tr, k)                  # <- selection sees training only
        m = make_model(seed)
        m.fit(X[np.ix_(tr, cols)], ly[tr])
        honest_pred[te] = m.predict(X[np.ix_(te, cols)])

    leaky, honest = _errors(y, leaky_pred), _errors(y, honest_pred)
    if verbose:
        print(f"leaky  selection (all cells):      {leaky['mean_pct_err']:.1f}% mean error")
        print(f"honest selection (in-fold only):   {honest['mean_pct_err']:.1f}% mean error")
        gap = honest["mean_pct_err"] - leaky["mean_pct_err"]
        print(f"optimism from selection leakage:   {gap:+.1f} percentage points")
        print(f"leaky picks: {[names[i] for i in leaky_cols]}")
    return {"leaky": leaky, "honest": honest,
            "leaky_features": [names[i] for i in leaky_cols]}


def main(cache: str | Path = "data/features.npz"):
    X, y, ids, names = load_cache(cache)
    print(f"{X.shape[0]} cells, {X.shape[1]} features, "
          f"cycle life {int(y.min())}-{int(y.max())}\n")

    print("=== paper split ===")
    evaluate_paper_split(X, y, ids, names)

    print("\n=== cross-validation across cells ===")
    evaluate_cv(X, y)

    print("\n=== leakage check ===")
    compare_leaky_vs_honest(X, y, names)


if __name__ == "__main__":
    main()
