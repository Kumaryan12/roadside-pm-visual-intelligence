import numpy as np

from pipelines.pm25_prediction.mumma_7day.run_nested_background_ensemble import (
    conformal_radius,
    group_crossfit_convex_predictions,
    optimal_convex_weight,
)


def test_optimal_convex_weight_prefers_exact_temporal_branch():
    truth = np.array([1.0, 2.0, 3.0])
    temporal = truth.copy()
    tabular = np.array([3.0, 3.0, 3.0])
    assert optimal_convex_weight(truth, temporal, tabular) == 1.0


def test_optimal_convex_weight_returns_half_for_identical_branches():
    branch = np.array([1.0, 2.0, 3.0])
    assert optimal_convex_weight(branch, branch, branch) == 0.5


def test_optimal_convex_weight_is_clipped():
    truth = np.array([10.0, 10.0])
    temporal = np.array([2.0, 2.0])
    tabular = np.array([1.0, 1.0])
    assert optimal_convex_weight(truth, temporal, tabular) == 1.0


def test_group_crossfit_convex_predictions_excludes_held_out_group():
    truth = np.array([0.0, 0.0, 10.0, 10.0])
    temporal = np.array([0.0, 0.0, 0.0, 0.0])
    tabular = np.array([10.0, 10.0, 10.0, 10.0])
    groups = np.array(["a", "a", "b", "b"])
    prediction = group_crossfit_convex_predictions(
        truth, temporal, tabular, groups,
    )
    np.testing.assert_allclose(prediction, np.array([10.0, 10.0, 0.0, 0.0]))


def test_conformal_radius_uses_finite_sample_higher_quantile():
    truth = np.arange(10, dtype=float)
    prediction = np.zeros(10, dtype=float)
    assert conformal_radius(truth, prediction, alpha=0.2) == 9.0
