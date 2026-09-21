"""Exponential decay model and curve fitting.

No Tkinter/matplotlib-GUI imports here on purpose: this module should be
usable from a plain script, a notebook, or a test file with nothing but
numpy/scipy installed.
"""
import numpy as np
from scipy.optimize import curve_fit


def exponential_decay(t, A, k, C):
    """I(t) = A * exp(-k*t) + C"""
    return A * np.exp(-k * t) + C


def fit_decay_constant(time_axis, intensity_profile):
    """Fit a single-exponential decay to a normalized intensity trace.

    Returns (A, k, C) as floats, or (nan, nan, nan) if fitting isn't
    possible (too few points, NaNs in the data, or the optimizer fails to
    converge within the given bounds).
    """
    try:
        if len(time_axis) < 4 or np.any(np.isnan(intensity_profile)):
            return np.nan, np.nan, np.nan

        plateau_guess = float(np.min(intensity_profile))
        amplitude_guess = float(intensity_profile[0] - plateau_guess)

        quarter_idx = max(1, len(time_axis) // 4)
        delta_t = time_axis[quarter_idx] - time_axis[0]
        delta_y = max(1e-4, intensity_profile[0] - intensity_profile[quarter_idx])
        k_guess = np.clip(delta_y / (delta_t * max(amplitude_guess, 1e-4)), 0.001, 5.0)

        p0 = [amplitude_guess, k_guess, plateau_guess]
        bounds = ([-2.0, 0.0, -1.0], [5.0, 50.0, 3.0])

        popt, _ = curve_fit(exponential_decay, time_axis, intensity_profile, p0=p0, bounds=bounds, maxfev=10000)
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return np.nan, np.nan, np.nan


def biexponential_decay(t, A1, k1, A2, k2, C):
    """I(t) = A1*exp(-k1*t) + A2*exp(-k2*t) + C"""
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + C


def fit_biexponential_decay(time_axis, intensity_profile):
    """Fit a two-exponential decay, both rates and both amplitudes free --
    nothing pinned from any other trace's fit.

    Returns (A1, k1, A2, k2, C) as floats, with k1 >= k2 by convention (the
    fast component first) so results are comparable across datasets instead
    of arbitrarily swapped -- the model is symmetric under relabeling
    (A1,k1)<->(A2,k2), so without a fixed convention "the fast term" would
    mean a different index every time. Returns a NaN 5-tuple if fitting
    isn't possible.
    """
    try:
        if len(time_axis) < 6 or np.any(np.isnan(intensity_profile)):
            return np.nan, np.nan, np.nan, np.nan, np.nan

        A, k, C = fit_decay_constant(time_axis, intensity_profile)
        if np.isnan(k):
            return np.nan, np.nan, np.nan, np.nan, np.nan

        # Seed as two copies of the single-exponential fit, split apart in
        # rate so the optimizer has somewhere to go rather than starting on
        # the degenerate A1=A2, k1=k2 ridge.
        p0 = [A / 2.0, k * 2.0, A / 2.0, k / 2.0, C]
        bounds = (
            [-2.0, 0.0, -2.0, 0.0, -1.0],
            [5.0, 50.0, 5.0, 50.0, 3.0],
        )

        popt, _ = curve_fit(
            biexponential_decay, time_axis, intensity_profile,
            p0=p0, bounds=bounds, maxfev=20000,
        )
        A1, k1, A2, k2, C_fit = (float(v) for v in popt)
        if k1 < k2:
            A1, k1, A2, k2 = A2, k2, A1, k1
        return A1, k1, A2, k2, C_fit
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def fit_biexponential_fixed_k1(time_axis, intensity_profile, k1_fixed):
    """Fit A1, A2, k2, C with k1 pinned to a rate already determined
    elsewhere -- normally the control trace's own single-exponential rate
    from fit_decay_constant -- instead of treating all five parameters as
    free the way fit_biexponential_decay does.

    Unlike the ratio-correction pipeline (intensity_traces.correct_with_control),
    this fits the RAW, uncorrected treated trace directly: the whole point of
    pinning k1 is to represent the imaging-induced component inside the same
    fit, so it should be called with a treated_norm trace, not a
    treated/control ratio.

    Returns a 5-tuple (A1, k1_fixed, A2, k2, C) -- the same shape
    fit_biexponential_decay returns, so it can be evaluated/plotted with the
    same biexponential_decay/evaluate_fit machinery. But k1 here is NOT
    sorted to be the larger rate the way fit_biexponential_decay's is -- it
    is whatever k1_fixed was given, which may be faster or slower than k2.
    Use fit_summary_fixed_k1, not fit_summary, to interpret the result.

    NaN 5-tuple if fitting isn't possible or k1_fixed itself is NaN (e.g.
    no control trace was available to fit).
    """
    try:
        if len(time_axis) < 5 or np.any(np.isnan(intensity_profile)) or np.isnan(k1_fixed):
            return np.nan, np.nan, np.nan, np.nan, np.nan

        def model(t, A1, A2, k2, C):
            return biexponential_decay(t, A1, k1_fixed, A2, k2, C)

        A_guess, k_guess, C_guess = fit_decay_constant(time_axis, intensity_profile)
        if np.isnan(k_guess):
            k_guess, C_guess = 0.1, 0.0
        p0 = [intensity_profile[0] / 2.0, intensity_profile[0] / 2.0, k_guess, C_guess]
        bounds = ([-2.0, -2.0, 0.0, -1.0], [5.0, 5.0, 50.0, 3.0])

        popt, _ = curve_fit(model, time_axis, intensity_profile, p0=p0, bounds=bounds, maxfev=20000)
        A1, A2, k2, C_fit = (float(v) for v in popt)
        return A1, float(k1_fixed), A2, k2, C_fit
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def fit_summary_fixed_k1(params):
    """Like fit_summary, for a fit_biexponential_fixed_k1 result: k1 here
    means "pinned to the control's rate," not "the faster component," so
    it isn't sorted or relabeled by magnitude the way fit_summary's
    two-exponential branch is.

    Returns a dict: model, k (=k2, the free/action-specific rate -- the
    comparable metric for pooling), k_imaging (=k1, fixed), k_action (=k2),
    label.
    """
    A1, k1, A2, k2, C = params
    return {"model": "Two-Exponential (fixed control k)", "k": k2,
            "k_imaging": k1, "k_action": k2,
            "label": f"k_imaging={k1:.4f} (fixed), k_action={k2:.4f} frame⁻¹"}


def fit_succeeded(params):
    """True if none of a fit's parameters are NaN -- works for either a
    3-tuple (single-exponential) or 5-tuple (two-exponential) result."""
    return not any(np.isnan(v) for v in params)


def evaluate_fit(t, params):
    """The fitted curve at t, dispatching on which model params belongs to
    (3 values = single-exponential, 5 = two-exponential). Centralizes that
    dispatch so a caller (a plot, a residual check) doesn't need to know the
    convention itself -- it can just hold onto whatever fit_decay_constant
    or fit_biexponential_decay handed back."""
    t = np.asarray(t, dtype=float)
    if len(params) == 3:
        return exponential_decay(t, *params)
    if len(params) == 5:
        return biexponential_decay(t, *params)
    raise ValueError(f"unrecognized fit params length: {len(params)}")


def fit_summary(params):
    """Interpret a fit's parameters for display/reporting, regardless of
    which model produced them.

    Returns a dict:
      model: "Single-Exponential" or "Two-Exponential"
      k: one representative rate, for the summary-CSV column every
         downstream consumer (pooling, spectral plots) already groups and
         averages on. For a two-exponential fit this is the slow
         component -- the fast component is a distinct phenomenon in its
         own right (see fit_biexponential_decay's docstring), not a detail
         to be averaged away, so it's also reported separately as k_fast.
      k_fast, k_slow: both two-exponential rates, or None for a
         single-exponential fit.
      label: a short string for a plot legend.
    """
    if len(params) == 3:
        A, k, C = params
        return {"model": "Single-Exponential", "k": k, "k_fast": None, "k_slow": None,
                "label": f"k={k:.4f} frame⁻¹"}
    if len(params) == 5:
        A1, k1, A2, k2, C = params
        return {"model": "Two-Exponential", "k": k2, "k_fast": k1, "k_slow": k2,
                "label": f"k_fast={k1:.4f}, k_slow={k2:.4f} frame⁻¹"}
    raise ValueError(f"unrecognized fit params length: {len(params)}")
