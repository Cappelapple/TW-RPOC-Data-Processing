import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from cell_kinetics.core import plotting


def make_traces(gfp_fit, mch_fit, gfp_fit_label=None, mch_fit_label=None):
    t = np.arange(20, dtype=float)
    traces = {
        "time_axis": t,
        "gfp_data": np.exp(-0.1 * t), "gfp_fit": gfp_fit,
        "has_gfp_control": True, "gfp_control": np.exp(-0.05 * t),
        "mch_data": np.exp(-0.2 * t), "mch_fit": mch_fit,
        "has_mch_control": True, "mch_control": np.exp(-0.05 * t),
        "borrowed_control": False,
    }
    if gfp_fit_label is not None:
        traces["gfp_fit_label"] = gfp_fit_label
    if mch_fit_label is not None:
        traces["mch_fit_label"] = mch_fit_label
    return traces


@pytest.mark.parametrize("fit_pair", [
    ((1.0, 0.1, 0.0), (1.0, 0.2, 0.0)),  # both single-exponential
    ((0.5, 0.3, 0.2, 0.05, 0.0), (0.6, 0.4, 0.3, 0.08, 0.0)),  # both two-exponential
    ((1.0, 0.1, 0.0), (0.6, 0.4, 0.3, 0.08, 0.0)),  # mixed: GFP single, mCherry two-exp
])
def test_build_current_fit_figure_handles_either_model(fit_pair):
    gfp_fit, mch_fit = fit_pair
    fig = plotting.build_current_fit_figure(make_traces(gfp_fit, mch_fit))
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_build_current_fit_figure_handles_failed_fit():
    traces = make_traces((np.nan, np.nan, np.nan), (1.0, 0.2, 0.0))
    fig = plotting.build_current_fit_figure(traces)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_build_individual_decay_figure_handles_either_model():
    traces = make_traces((0.5, 0.3, 0.2, 0.05, 0.0), (1.0, 0.2, 0.0))
    fig = plotting.build_individual_decay_figure("dataset_x", traces)
    assert fig is not None
    matplotlib.pyplot.close(fig)


def test_explicit_fit_label_overrides_generic_summary_for_fixed_k1():
    # A fixed-k1 result where k1 (0.02, the "imaging" rate) is slower than
    # k2 (0.5, the free "action" rate) -- generic fit_summary would mislabel
    # this as k_fast=0.02/k_slow=0.5, which is backwards for a fixed-k1 fit.
    # The explicit gfp_fit_label must be what actually shows up in the legend.
    gfp_fit = (0.4, 0.02, 0.35, 0.5, 0.1)
    explicit_label = "k_imaging=0.0200 (fixed), k_action=0.5000 frame⁻¹"
    traces = make_traces(gfp_fit, (1.0, 0.2, 0.0), gfp_fit_label=explicit_label)

    fig = plotting.build_current_fit_figure(traces)
    legend_texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert explicit_label in legend_texts
    assert not any("k_fast=0.0200" in t for t in legend_texts)
    matplotlib.pyplot.close(fig)
