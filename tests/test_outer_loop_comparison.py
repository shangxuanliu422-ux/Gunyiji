"""Check analytic controller mathematics against the existing constrained solver."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_outer_loop_comparison import (
    CasadiPositionMPC, FeedforwardLQR, FeedforwardPD, IncrementalPreviewLQT,
)


def test_incremental_tracking_matches_mpc_when_bounds_inactive():
    horizon = 12
    ref = np.zeros((horizon + 1, 9))
    t = np.arange(horizon + 1) * .1
    ref[:, 0] = .02 * np.sin(t)
    ref[:, 3] = .02 * np.cos(t)
    ref[:, 6] = -.02 * np.sin(t)
    state = ref[0, :6].copy()
    state[:3] += [.005, -.003, .002]
    previous = np.array([.01, -.01, .01])
    mpc = CasadiPositionMPC(.1, horizon).solve(state, previous, ref)
    lqt = IncrementalPreviewLQT(horizon, terminal="mpc").solve(state, previous, ref)
    assert mpc.success
    assert np.max(np.abs(mpc.predicted_accelerations)) < 1.
    np.testing.assert_allclose(lqt.acceleration, mpc.acceleration, atol=2e-5)


def test_lqr_gains_stabilize_design_models():
    assert FeedforwardLQR().spectral_radius < 1.
    assert IncrementalPreviewLQT().spectral_radius < 1.


def test_feedforward_and_output_limits():
    ref = np.zeros((5, 9)); ref[:, 6:9] = [.2, -.3, .4]
    for ctrl in (FeedforwardLQR(), FeedforwardPD()):
        np.testing.assert_allclose(ctrl.solve(np.zeros(6), np.zeros(3), ref).acceleration, ref[0,6:9])
        output = ctrl.solve(np.array([100.,-100.,100.,0.,0.,0.]),np.zeros(3),ref).acceleration
        assert np.all(np.abs(output) <= [2.,2.,3.])
