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
