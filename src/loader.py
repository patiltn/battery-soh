"""
Loader for the Severson et al. (2019) fast-charging dataset.

The raw files are MATLAB v7.3, i.e. HDF5 underneath, so `scipy.io.loadmat`
cannot read them.  Every field is stored as an array of HDF5 object
references that has to be dereferenced one at a time -- `f[ref][:]` -- which
is why this is a module rather than three lines.

File structure (verified against 2017-06-30, 48 cells):

    batch/
      barcode         (n_cells, 1) refs -> uint16 char codes
      channel_id      (n_cells, 1)
      cycle_life      (n_cells, 1) refs -> (1,1) float
      policy_readable (n_cells, 1) refs -> char codes
      Vdlin           (n_cells, 1) refs -> (1000,) voltage grid
      summary         (n_cells, 1) refs -> struct of (1, n_cycles) arrays:
                          IR QCharge QDischarge Tavg Tmax Tmin chargetime cycle
      cycles          (n_cells, 1) refs -> struct of (n_cycles, 1) ref arrays:
                          I Qc Qd Qdlin T Tdlin V discharge_dQdV t

`Qdlin` is the dataset's own discharge capacity interpolated onto the fixed
`Vdlin` grid -- the same object `features.q_of_v` constructs from raw V/Qd.
Both are loaded so the two can be cross-checked against each other; agreeing
with the published interpolation is a useful check that our own is right.

Data cleaning below follows the authors' own `Load Data.ipynb` in
github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
rather than any judgement of ours.  These are not optional: the stitching in
particular changes the prediction TARGET for five cells, and skipping it
silently trains the model on cycle lives that are wrong by hundreds of cycles.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

BATCH_FILES = {
    1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

# Cells that never reach 80% capacity (batch 1) or sit on noisy channels
# (batch 3).  Authors' exclusions, reproduced verbatim.
EXCLUDE = {
    1: ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"],
    2: [],
    3: ["b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"],
}

# Five batch-1 cells had their tests continued in batch 2 under new keys.
# Their real cycle life is the sum of both runs; the batch-2 entries are
# duplicates of the same physical cells and are removed after merging.
CONTINUATION = {
    "b1c0": ("b2c7", 662),
    "b1c1": ("b2c8", 981),
    "b1c2": ("b2c9", 1060),
    "b1c3": ("b2c15", 208),
    "b1c4": ("b2c16", 482),
}

SUMMARY_FIELDS = ["IR", "QCharge", "QDischarge", "Tavg", "Tmax", "Tmin",
                  "chargetime", "cycle"]
CYCLE_FIELDS = ["I", "Qc", "Qd", "Qdlin", "T", "Tdlin", "V",
                "discharge_dQdV", "t"]


def _chars(f: h5py.File, ref) -> str:
    """MATLAB char arrays arrive as uint16 code points."""
    return "".join(chr(int(c)) for c in np.asarray(f[ref][:]).flatten() if 32 <= int(c) < 1114111)


def load_batch(path: str | os.PathLike, batch_id: int,
               max_cycle: int = 100,
               cycle_fields: list[str] | None = None) -> dict:
    """Load one batch file.

    Only cycles 1..`max_cycle` are read.  This is not just a speed
    optimisation: the whole premise is early-cycle prediction, so reading
    later cycles into the same structure would make it easy to use them by
    accident.  Set `max_cycle=None` to read everything (slow, ~GB of RAM).

    Returns {cell_key: {cycle_life, policy, barcode, channel, Vdlin,
                        summary: {field: array}, cycles: {idx: {field: array}}}}
    """
    cycle_fields = cycle_fields or CYCLE_FIELDS
    path = Path(path)
    out: dict[str, dict] = {}

    with h5py.File(path, "r") as f:
        b = f["batch"]
        n_cells = b["summary"].shape[0]

        for i in range(n_cells):
            key = f"b{batch_id}c{i}"

            cl = f[b["cycle_life"][i, 0]][:]
            cycle_life = float(cl[0, 0]) if cl.size else np.nan

            summary_grp = f[b["summary"][i, 0]]
            summary = {fld: np.asarray(summary_grp[fld][:]).flatten()
                       for fld in SUMMARY_FIELDS if fld in summary_grp}

            cyc_grp = f[b["cycles"][i, 0]]
            n_cycles = cyc_grp["I"].shape[0]
            last = n_cycles if max_cycle is None else min(n_cycles, max_cycle)

            cycles: dict[int, dict] = {}
            for j in range(last):
                rec = {}
                for fld in cycle_fields:
                    if fld in cyc_grp:
                        rec[fld] = np.asarray(f[cyc_grp[fld][j, 0]][:]).flatten()
                # cycle numbering is 1-based to match `summary['cycle']`
                cycles[j + 1] = rec

            out[key] = {
                "cycle_life": cycle_life,
                "policy": _chars(f, b["policy_readable"][i, 0]),
                "barcode": _chars(f, b["barcode"][i, 0]),
                "channel": int(f[b["channel_id"][i, 0]][0, 0]),
                "Vdlin": np.asarray(f[b["Vdlin"][i, 0]][:]).flatten(),
                "summary": summary,
                "cycles": cycles,
            }
    return out


def _stitch(batch1: dict, batch2: dict) -> None:
    """Merge continued tests from batch 2 back into their batch-1 cells.

    Modifies both dicts in place: batch-1 cycle lives and summaries are
    extended, and the duplicate batch-2 entries are deleted.
    """
    for b1_key, (b2_key, add_len) in CONTINUATION.items():
        if b1_key not in batch1 or b2_key not in batch2:
            continue
        batch1[b1_key]["cycle_life"] += add_len

        s1, s2 = batch1[b1_key]["summary"], batch2[b2_key]["summary"]
        offset = len(s1.get("cycle", []))
        for fld in s1:
            if fld not in s2:
                continue
            tail = s2[fld] + offset if fld == "cycle" else s2[fld]
            s1[fld] = np.hstack([s1[fld], tail])

        del batch2[b2_key]


def load_dataset(raw_dir: str | os.PathLike = "data/raw",
                 batches: tuple[int, ...] = (1, 2, 3),
                 max_cycle: int = 100,
                 apply_cleaning: bool = True) -> dict:
    """Load the batches present in `raw_dir`, cleaned and stitched.

    Missing batch files are skipped with a warning rather than raising, so
    the pipeline can be developed against whichever batches have finished
    downloading.  Note that stitching only happens if BOTH batch 1 and 2 are
    loaded -- otherwise five cells keep truncated cycle lives, so a run on
    batch 1 alone is for development only, never for reported results.
    """
    raw_dir = Path(raw_dir)
    loaded: dict[int, dict] = {}

    for bid in batches:
        p = raw_dir / BATCH_FILES[bid]
        if not p.exists():
            print(f"[load_dataset] missing, skipping: {p.name}")
            continue
        loaded[bid] = load_batch(p, bid, max_cycle=max_cycle)
        print(f"[load_dataset] batch {bid}: {len(loaded[bid])} cells")

    if apply_cleaning:
        if 1 in loaded and 2 in loaded:
            _stitch(loaded[1], loaded[2])
        elif 1 in loaded:
            print("[load_dataset] WARNING: batch 2 absent -- 5 batch-1 cells "
                  "keep truncated cycle lives. Development only.")
        for bid, cells in loaded.items():
            for key in EXCLUDE[bid]:
                cells.pop(key, None)

    merged: dict[str, dict] = {}
    for bid in sorted(loaded):
        merged.update(loaded[bid])
    print(f"[load_dataset] total cells after cleaning: {len(merged)}")
    return merged


def to_feature_input(cell: dict) -> tuple[dict, dict]:
    """Adapt one loaded cell to the (cycles, summary) shape `features` expects.

    `features.cell_features` works on raw V/Qd/I/t/T per cycle plus a summary
    with `cycle` and `QD`; the raw file calls the latter `QDischarge`.
    """
    cycles = {
        idx: {"V": rec["V"], "Qd": rec["Qd"], "I": rec["I"],
              "t": rec["t"], "T": rec["T"]}
        for idx, rec in cell["cycles"].items()
        if all(k in rec for k in ("V", "Qd", "I", "t", "T"))
    }
    summary = {
        "cycle": np.asarray(cell["summary"]["cycle"], dtype=float),
        "QD": np.asarray(cell["summary"]["QDischarge"], dtype=float),
    }
    return cycles, summary


def cache_features(raw_dir: str | os.PathLike = "data/raw",
                   out_path: str | os.PathLike = "data/features.npz",
                   max_cycle: int = 100, early: int = 10):
    """Extract features for every cell and save a small cache.

    The raw files are ~8 GB; this cache is a few MB and is all the modelling
    stage needs, so the raw data can be deleted (or moved off the machine)
    once this has run.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import cell_features

    data = load_dataset(raw_dir, max_cycle=max_cycle)
    rows, lives, ids, policies = [], [], [], []
    for key, cell in sorted(data.items()):
        cycles, summary = to_feature_input(cell)
        try:
            rows.append(cell_features(cycles, summary, early=early,
                                      late=max_cycle))
        except (ValueError, KeyError) as e:
            print(f"[cache_features] skipping {key}: {e}")
            continue
        lives.append(cell["cycle_life"])
        ids.append(key)
        policies.append(cell["policy"])

    names = sorted({k for r in rows for k in r})
    X = np.array([[r.get(n, np.nan) for n in names] for r in rows])

    # Drop features that are NaN for EVERY cell.  On this dataset that is the
    # second incremental-capacity peak (IC1_*, dIC1_*): at the 4C discharge
    # rate used here, polarisation smears the LFP plateaus together so only
    # one peak is resolved.  The feature is not missing, it is absent -- there
    # is no second peak to measure -- so imputing a value would be inventing
    # data. Columns are dropped rather than filled, and the names are printed
    # so the loss is visible rather than silent.
    all_nan = np.isnan(X).all(axis=0)
    if all_nan.any():
        dropped = [n for n, d in zip(names, all_nan) if d]
        print(f"[cache_features] dropping all-NaN features: {', '.join(dropped)}")
        X = X[:, ~all_nan]
        names = [n for n, d in zip(names, all_nan) if not d]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=np.array(lives, dtype=float),
                        ids=np.array(ids), policies=np.array(policies),
                        feature_names=np.array(names))
    print(f"[cache_features] wrote {out_path} -- {X.shape[0]} cells, "
          f"{X.shape[1]} features")
    return out_path


if __name__ == "__main__":
    cache_features()
