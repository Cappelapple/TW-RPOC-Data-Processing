import numpy as np
import pytest

from cell_kinetics.core.fitting import (
    biexponential_decay,
    evaluate_fit,
    exponential_decay,
    fit_biexponential_decay,
    fit_biexponential_fixed_k1,
    fit_decay_constant,
    fit_succeeded,
    fit_summary,
    fit_summary_fixed_k1,
)


def test_exponential_decay_known_values():
    # A=1, k=1, C=0 at t=0 -> 1.0; as t -> large, approaches C.
    assert np.isclose(exponential_decay(0.0, A=1.0, k=1.0, C=0.0), 1.0)
    assert exponential_decay(50.0, A=1.0, k=1.0, C=0.2) < 0.21


def test_fit_recovers_known_parameters_noiseless():
    t = np.arange(20, dtype=float)
    true_A, true_k, true_C = 0.7, 0.15, 0.2
    y = exponential_decay(t, true_A, true_k, true_C)

    A, k, C = fit_decay_constant(t, y)

    assert np.isclose(A, true_A, atol=1e-3)
    assert np.isclose(k, true_k, atol=1e-3)
    assert np.isclose(C, true_C, atol=1e-3)


def test_fit_handles_short_series():
    # Fewer than 4 points: fit_decay_constant should bail out cleanly, not raise.
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 0.8, 0.6])
    A, k, C = fit_decay_constant(t, y)
    assert np.isnan(A) and np.isnan(k) and np.isnan(C)


def test_fit_handles_nan_input():
    t = np.arange(10, dtype=float)
    y = np.full(10, np.nan)
    A, k, C = fit_decay_constant(t, y)
    assert np.isnan(A) and np.isnan(k) and np.isnan(C)


def test_biexponential_recovers_known_parameters_noiseless():
    t = np.arange(40, dtype=float)
    true = (0.5, 0.4, 0.3, 0.05, 0.1)  # A1, k1, A2, k2, C -- k1 > k2
    y = biexponential_decay(t, *true)

    A1, k1, A2, k2, C = fit_biexponential_decay(t, y)

    assert np.isclose(A1, true[0], atol=1e-2)
    assert np.isclose(k1, true[1], atol=1e-2)
    assert np.isclose(A2, true[2], atol=1e-2)
    assert np.isclose(k2, true[3], atol=1e-2)
    assert np.isclose(C, true[4], atol=1e-2)


def test_biexponential_fast_component_always_reported_first():
    t = np.arange(40, dtype=float)
    # Seed the "true" fast/slow rates swapped relative to how they're passed
    # in -- k1 < k2 here -- to check the convention swaps them back.
    y = biexponential_decay(t, 0.3, 0.05, 0.5, 0.4, 0.1)

    A1, k1, A2, k2, C = fit_biexponential_decay(t, y)
    assert k1 >= k2


def test_biexponential_handles_short_series():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 0.8, 0.6, 0.5, 0.4])
    A1, k1, A2, k2, C = fit_biexponential_decay(t, y)
    assert all(np.isnan(v) for v in (A1, k1, A2, k2, C))


def test_biexponential_handles_nan_input():
    t = np.arange(10, dtype=float)
    y = np.full(10, np.nan)
    result = fit_biexponential_decay(t, y)
    assert all(np.isnan(v) for v in result)


def test_fit_succeeded_true_for_clean_params():
    assert fit_succeeded((1.0, 0.2, 0.1))
    assert fit_succeeded((1.0, 0.2, 0.3, 0.05, 0.1))


def test_fit_succeeded_false_when_any_nan():
    assert not fit_succeeded((np.nan, np.nan, np.nan))
    assert not fit_succeeded((1.0, np.nan, 0.3, 0.05, 0.1))


def test_evaluate_fit_dispatches_on_param_count():
    t = np.arange(10, dtype=float)
    single = evaluate_fit(t, (1.0, 0.2, 0.1))
    assert np.allclose(single, exponential_decay(t, 1.0, 0.2, 0.1))

    bi = evaluate_fit(t, (0.5, 0.4, 0.3, 0.05, 0.1))
    assert np.allclose(bi, biexponential_decay(t, 0.5, 0.4, 0.3, 0.05, 0.1))


def test_evaluate_fit_rejects_unknown_param_count():
    with pytest.raises(ValueError):
        evaluate_fit(np.arange(5, dtype=float), (1.0, 2.0))


def test_fit_summary_single_exponential():
    summary = fit_summary((1.0, 0.25, 0.1))
    assert summary["model"] == "Single-Exponential"
    assert summary["k"] == 0.25
    assert summary["k_fast"] is None
    assert summary["k_slow"] is None


def test_fit_summary_two_exponential_reports_slow_as_k():
    summary = fit_summary((0.5, 0.4, 0.3, 0.05, 0.1))
    assert summary["model"] == "Two-Exponential"
    assert summary["k"] == 0.05
    assert summary["k_fast"] == 0.4
    assert summary["k_slow"] == 0.05


def test_fixed_k1_recovers_known_free_parameters_noiseless():
    t = np.arange(40, dtype=float)
    true = (0.4, 0.05, 0.35, 0.3, 0.1)  # A1, k1(fixed), A2, k2(free), C
    y = biexponential_decay(t, *true)

    A1, k1, A2, k2, C = fit_biexponential_fixed_k1(t, y, k1_fixed=true[1])

    assert k1 == true[1]  # pinned, not fit
    assert np.isclose(A1, true[0], atol=1e-2)
    assert np.isclose(A2, true[2], atol=1e-2)
    assert np.isclose(k2, true[3], atol=1e-2)
    assert np.isclose(C, true[4], atol=1e-2)


def test_fixed_k1_does_not_sort_by_magnitude():
    # k1_fixed is deliberately the SLOWER rate here -- fit_biexponential_decay
    # would swap this to put the faster rate first; fixed_k1 must not, since
    # k1's identity (the control's own rate) is what's meaningful, not its
    # relative size.
    t = np.arange(40, dtype=float)
    true = (0.4, 0.02, 0.35, 0.5, 0.1)  # k1=0.02 (slow, fixed), k2=0.5 (fast, free)
    y = biexponential_decay(t, *true)

    A1, k1, A2, k2, C = fit_biexponential_fixed_k1(t, y, k1_fixed=true[1])
    assert k1 == true[1]
    assert np.isclose(k2, true[3], atol=1e-2)


def test_fixed_k1_nan_when_k1_fixed_is_nan():
    t = np.arange(40, dtype=float)
    y = np.exp(-0.1 * t)
    result = fit_biexponential_fixed_k1(t, y, k1_fixed=np.nan)
    assert all(np.isnan(v) for v in result)


def test_fixed_k1_handles_short_series():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 0.8, 0.6, 0.5])
    result = fit_biexponential_fixed_k1(t, y, k1_fixed=0.1)
    assert all(np.isnan(v) for v in result)


def test_fit_summary_fixed_k1_reports_k2_as_k():
    summary = fit_summary_fixed_k1((0.4, 0.02, 0.35, 0.5, 0.1))
    assert summary["model"] == "Two-Exponential (fixed control k)"
    assert summary["k"] == 0.5
    assert summary["k_imaging"] == 0.02
    assert summary["k_action"] == 0.5
