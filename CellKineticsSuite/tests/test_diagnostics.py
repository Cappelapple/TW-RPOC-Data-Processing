import numpy as np

from cell_kinetics.core.diagnostics import aic, bic, fit_residuals, r_squared, runs_test_pvalue
from cell_kinetics.core.fitting import exponential_decay


def test_fit_residuals_zero_for_exact_fit():
    t = np.arange(20, dtype=float)
    params = (2.0, 0.3, 0.1)
    data = exponential_decay(t, *params)
    residuals = fit_residuals(t, data, params)
    assert np.allclose(residuals, 0.0, atol=1e-10)


def test_fit_residuals_nan_when_fit_failed():
    t = np.arange(10, dtype=float)
    residuals = fit_residuals(t, np.ones(10), (np.nan, np.nan, np.nan))
    assert np.all(np.isnan(residuals))


def test_runs_test_flags_systematic_pattern_more_than_noise_like_pattern():
    # A slow hump: residuals stay one sign for long stretches -- the
    # signature of a model that's systematically off, not just noisy.
    t = np.linspace(0, 2 * np.pi, 40)
    systematic = np.sin(t)

    # Actual random noise (fixed seed, so the test is deterministic): sign
    # changes at whatever rate chance produces, not the "hump" pattern above
    # and not artificial every-single-point alternation either -- both of
    # those are themselves non-random in opposite directions.
    noise_like = np.random.RandomState(0).normal(size=40)

    p_systematic = runs_test_pvalue(systematic)
    p_noise_like = runs_test_pvalue(noise_like)

    assert p_systematic < p_noise_like
    assert p_systematic < 0.05


def test_runs_test_all_same_sign_is_maximally_non_random():
    assert runs_test_pvalue(np.full(10, 1.0)) == 0.0


def test_runs_test_nan_with_no_data():
    assert np.isnan(runs_test_pvalue(np.array([])))


def test_r_squared_perfect_fit_is_one():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    residuals = np.zeros(4)
    assert r_squared(data, residuals) == 1.0


def test_r_squared_nan_when_residuals_nan():
    data = np.array([1.0, 2.0, 3.0])
    residuals = np.array([np.nan, 0.1, 0.2])
    assert np.isnan(r_squared(data, residuals))


def test_aic_prefers_lower_rss_at_equal_params():
    tight = np.full(20, 0.01)
    loose = np.full(20, 0.5)
    assert aic(tight, n_params=3) < aic(loose, n_params=3)


def test_aic_penalizes_extra_params_at_equal_rss():
    residuals = np.full(20, 0.1)
    assert aic(residuals, n_params=3) < aic(residuals, n_params=5)


def test_bic_penalizes_extra_params_more_than_aic_does():
    residuals = np.full(50, 0.1)
    aic_gap = aic(residuals, n_params=5) - aic(residuals, n_params=3)
    bic_gap = bic(residuals, n_params=5) - bic(residuals, n_params=3)
    assert bic_gap > aic_gap


def test_aic_nan_when_residuals_contain_nan():
    assert np.isnan(aic(np.array([0.1, np.nan]), n_params=3))
