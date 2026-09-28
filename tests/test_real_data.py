"""
Validation against the published dataset.

The tests in `test_features.py` check our implementation against a simulator
we wrote ourselves, which can only catch internal inconsistency: if the same
misconception is in both, both agree and both are wrong.  These tests check
it against an INDEPENDENT implementation -- the `Qdlin` curves shipped with
the Severson dataset, produced by the original authors from the same raw
traces.  Agreement there is evidence the feature layer is actually right.

They are skipped automatically when the raw `.mat` files are absent, so the
suite still runs on a clean clone (and in CI) without the 8 GB download.

Run:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import dqdv, q_of_v  # noqa: E402
from loader import BATCH_FILES, load_dataset  # noqa: E402

RAW = ROOT / "data" / "raw"
# Batch 2 is the reference here simply because it is self-contained: no cells
# are excluded from it and none are stitched, so any disagreement is about the
# curve maths rather than the cleaning.
BATCH2 = RAW / BATCH_FILES[2]

pytestmark = pytest.mark.skipif(
    not BATCH2.exists(),
    reason=f"raw data not present ({BATCH2.name}); download to run",
)

# The extreme ends of the voltage grid are where the discharge is sparsely
# sampled and where any two implementations differ by how they treat the
# boundary (clip vs extrapolate vs spline).  Comparing there tests the edge
# convention, not the method, so the interior is what we assert on.
EDGE_TRIM = 20
TOL_MEDIAN = 0.005      # Ah
TOL_P99 = 0.010         # Ah


@pytest.fixture(scope="module")
def batch2():
    return load_dataset(RAW, batches=(2,), max_cycle=100)


def _compare(cell, cycle_idx):
    """Our Q(V) vs the published Qdlin, on the cell's own Vdlin grid."""
    rec = cell["cycles"][cycle_idx]
    mine = q_of_v(rec["V"], rec["Qd"], np.asarray(cell["Vdlin"]), I=rec["I"])
    theirs = np.asarray(rec["Qdlin"])
    sl = slice(EDGE_TRIM, -EDGE_TRIM)
    m = np.isfinite(mine[sl]) & np.isfinite(theirs[sl])
    return np.abs(mine[sl][m] - theirs[sl][m])


def test_q_of_v_matches_published_qdlin(batch2):
    """Our interpolation must agree with the authors' in the curve interior."""
    cell = batch2[sorted(batch2)[0]]
    diff = _compare(cell, 100)
    assert diff.size > 500
    assert np.median(diff) < TOL_MEDIAN
    assert np.percentile(diff, 99) < TOL_P99


def test_q_of_v_agreement_holds_across_cells(batch2):
    """Agreement is a property of the method, not of one lucky cell."""
    medians = []
    for key in sorted(batch2)[:10]:
        cell = batch2[key]
        if 100 not in cell["cycles"]:
            continue
        medians.append(float(np.median(_compare(cell, 100))))
    assert len(medians) >= 5
    assert max(medians) < TOL_MEDIAN


def test_discharge_masking_improves_agreement(batch2):
    """Masking to the discharge branch must move us CLOSER to the published curve.

    This is the test that pins the fix in place: a raw cycle contains both
    charge and discharge, and interpolating over both blends two physically
    distinct curves.  If someone later drops the `I=` argument, this fails.
    """
    cell = batch2[sorted(batch2)[0]]
    rec = cell["cycles"][100]
    vd = np.asarray(cell["Vdlin"])
    theirs = np.asarray(rec["Qdlin"])
    sl = slice(EDGE_TRIM, -EDGE_TRIM)

    masked = q_of_v(rec["V"], rec["Qd"], vd, I=rec["I"])
    unmasked = q_of_v(rec["V"], rec["Qd"], vd)
    m = np.isfinite(masked[sl]) & np.isfinite(unmasked[sl]) & np.isfinite(theirs[sl])

    err_masked = np.abs(masked[sl][m] - theirs[sl][m]).max()
    err_unmasked = np.abs(unmasked[sl][m] - theirs[sl][m]).max()
    assert err_masked < err_unmasked


def test_capacity_endpoint_matches_summary(batch2):
    """Q(V) at the low-voltage end must match the summary's discharge capacity.

    An independent check on the curve's overall SCALE: the two come from
    different fields of the file, so agreement means we have not mangled units
    or picked up the charge branch's capacity by mistake.
    """
    cell = batch2[sorted(batch2)[0]]
    rec = cell["cycles"][100]
    mine = q_of_v(rec["V"], rec["Qd"], np.asarray(cell["Vdlin"]), I=rec["I"])
    q_end = np.nanmax(mine)
    q_summary = float(cell["summary"]["QDischarge"][99])
    assert abs(q_end - q_summary) / q_summary < 0.05


def test_dqdv_on_real_data_has_lfp_peaks(batch2):
    """Real LFP cells must show incremental-capacity peaks in the 3.2-3.4 V band."""
    cell = batch2[sorted(batch2)[0]]
    rec = cell["cycles"][100]
    ic = dqdv(rec["V"], rec["Qd"], np.asarray(cell["Vdlin"]), I=rec["I"])
    vd = np.asarray(cell["Vdlin"])
    band = (vd > 3.1) & (vd < 3.45)
    valid = np.isfinite(ic)
    assert (band & valid).sum() > 50
    assert np.nanmax(ic[band & valid]) > np.nanmedian(ic[valid])


def test_cleaning_produces_expected_cell_count():
    """Batch 1 + 2, cleaned and stitched, must give 84 cells.

    46 + 48 = 94, minus 5 batch-1 cells that never reach 80% capacity, minus
    the 5 batch-2 duplicates of tests continued from batch 1.
    """
    if not (RAW / BATCH_FILES[1]).exists():
        pytest.skip("batch 1 not present")
    d = load_dataset(RAW, batches=(1, 2), max_cycle=10)
    assert len(d) == 84
    assert "b2c7" not in d and "b1c8" not in d
    assert d["b1c0"]["cycle_life"] > 1000    # stitched, not truncated
