import numpy as np

from cell_kinetics.core.fitting import exponential_decay, fit_decay_constant


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
