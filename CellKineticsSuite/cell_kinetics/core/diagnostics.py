"""Checking whether a fitted model actually explains a trace, rather than
just trusting that curve_fit converged.

The specific question this exists to answer: is the ratio-corrected treated
trace (see intensity_traces.correct_with_control) genuinely single-exponential,
or is there leftover structure a single exponential can't capture? If the
residuals are noise-like (no autocorrelation), the additive-bleaching-rate
assumption behind the ratio method holds and there's no need for a more
complex model. If they show a systematic pattern (a hump, a trend), that's
real evidence the single-exponential model is missing something.
"""
import numpy as np
from scipy.stats import norm

from .fitting import exponential_decay


def fit_residuals(time_axis, data, params):
    """data - exponential_decay(time_axis, *params). NaN array if params
    contains NaN (i.e. the fit didn't converge)."""
    A, k, C = params
    data = np.asarray(data, dtype=float)
    if np.isnan(A) or np.isnan(k) or np.isnan(C):
        return np.full_like(data, np.nan)
    model = exponential_decay(np.asarray(time_axis, dtype=float), A, k, C)
    return data - model


def runs_test_pvalue(residuals):
    """Wald-Wolfowitz runs test on the sign of residuals around zero.

    A well-fit model leaves residuals whose signs are randomly ordered (no
    run of consecutive same-sign points longer than chance). A low p-value
    means the signs are NOT randomly ordered -- e.g. all the early residuals
    are positive and all the late ones are negative -- which is the signature
    of systematic curvature the model didn't capture, not just noise.

    Returns NaN if there are too few non-zero residuals to test (fewer than
    two distinct signs).
    """
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    signs = np.sign(r)
    signs = signs[signs != 0]

    n1 = int(np.sum(signs > 0))
    n2 = int(np.sum(signs < 0))
    if n1 == 0 or n2 == 0:
        return np.nan if (n1 + n2) == 0 else 0.0

    n = n1 + n2
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    mean_runs = 2.0 * n1 * n2 / n + 1.0
    var_runs = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n * n * (n - 1))
    if var_runs <= 0:
        return np.nan

    z = (runs - mean_runs) / np.sqrt(var_runs)
    return float(2.0 * (1.0 - norm.cdf(abs(z))))


def aic(residuals, n_params):
    """Akaike Information Criterion under i.i.d. Gaussian-noise residuals:
    AIC = n*ln(RSS/n) + 2*n_params. Lower is better. Penalizes extra
    parameters, so this is what makes "the 2-exponential fit had lower RSS"
    a fair comparison instead of a foregone conclusion -- a model with more
    free parameters can always fit at least as well by construction, so RSS
    or R^2 alone can't tell you whether the extra complexity earned its
    keep. NaN if residuals contains NaN (fit didn't converge) or is empty.
    """
    r = np.asarray(residuals, dtype=float)
    if r.size == 0 or np.any(np.isnan(r)):
        return np.nan
    n = r.size
    rss = float(np.sum(r ** 2))
    if rss <= 0:
        return -np.inf
    return n * np.log(rss / n) + 2.0 * n_params


def bic(residuals, n_params):
    """Bayesian Information Criterion: like aic(), but penalizes extra
    parameters more heavily as n grows (2*n_params -> n_params*ln(n)).
    Lower is better."""
    r = np.asarray(residuals, dtype=float)
    if r.size == 0 or np.any(np.isnan(r)):
        return np.nan
    n = r.size
    rss = float(np.sum(r ** 2))
    if rss <= 0:
        return -np.inf
    return n * np.log(rss / n) + n_params * np.log(n)


def r_squared(data, residuals):
    """Fraction of variance in data explained by the fit. NaN if residuals
    contains NaN or data is constant (zero variance, R^2 undefined)."""
    data = np.asarray(data, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if np.any(np.isnan(residuals)):
        return np.nan
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((data - np.mean(data)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot
