"""Is the ratio-corrected treated trace actually single-exponential?

Run against a results folder (the one with session_traces_cache.pkl in it)
to check every already-processed dataset's fit for leftover structure the
single exponential didn't capture. Prints a table sorted worst-first (lowest
runs-test p-value = most likely non-random residuals = the model is missing
something), and saves a PNG of the worst few and best few fits so you can
look at them directly.

Usage:
    py -3.12 tools/check_single_exponential_fit.py "<results folder>"
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cell_kinetics.core import diagnostics, persistence  # noqa: E402


def build_report(traces_cache):
    rows = []
    for dataset_name, entry in traces_cache.items():
        time_axis = entry["time_axis"]
        for channel, data_key, fit_key in (("GFP", "gfp_data", "gfp_fit"), ("mCherry", "mch_data", "mch_fit")):
            data = entry.get(data_key)
            params = entry.get(fit_key)
            if data is None or params is None:
                continue
            residuals = diagnostics.fit_residuals(time_axis, data, params)
            rows.append({
                "Dataset": dataset_name,
                "Channel": channel,
                "k": params[1],
                "R2": diagnostics.r_squared(data, residuals),
                "runs_test_p": diagnostics.runs_test_pvalue(residuals),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("runs_test_p", na_position="last").reset_index(drop=True)


def plot_case(ax_top, ax_bottom, dataset_name, channel, time_axis, data, params):
    residuals = diagnostics.fit_residuals(time_axis, data, params)
    model = data - residuals

    ax_top.plot(time_axis, data, "o", ms=3, color="#2c3e50", label="data")
    ax_top.plot(time_axis, model, "-", color="#e74c3c", label="single-exp fit")
    ax_top.set_title(f"{dataset_name} ({channel})", fontsize=9)
    ax_top.legend(fontsize=7)

    ax_bottom.axhline(0, color="#7f8c8d", lw=1)
    ax_bottom.plot(time_axis, residuals, "o", ms=3, color="#8e44ad")
    ax_bottom.set_ylabel("residual", fontsize=8)


def main():
    if len(sys.argv) < 2:
        print("Usage: py -3.12 tools/check_single_exponential_fit.py \"<results folder>\"")
        sys.exit(1)

    folder = sys.argv[1]
    traces_cache, _ = persistence.load_session_cache(folder)
    if not traces_cache:
        print(f"No {persistence.CACHE_FILENAME} found (or it's empty) in: {folder}")
        print("Process at least one dataset in the app first -- this reads what it already fit and cached.")
        sys.exit(1)

    report = build_report(traces_cache)
    if report.empty:
        print("Cache had entries but none had both a trace and a fit to check.")
        sys.exit(1)

    pd.set_option("display.width", 120)
    pd.set_option("display.max_rows", None)
    print(report.to_string(index=False))

    n_significant = int((report["runs_test_p"] < 0.05).sum())
    n_tested = int(report["runs_test_p"].notna().sum())
    print(f"\n{n_significant}/{n_tested} fits have non-random residuals at p<0.05")
    print("(that's evidence the single-exponential model is missing structure for those -- worth")
    print(" looking at the plot before assuming a fixed-rate bi-exponential would fix it)")

    worst = report.head(4)
    best = report.dropna(subset=["runs_test_p"]).tail(4)
    cases = pd.concat([worst, best]).drop_duplicates(subset=["Dataset", "Channel"])
    if cases.empty:
        return

    fig, axes = plt.subplots(2, len(cases), figsize=(3.2 * len(cases), 5), squeeze=False)
    for col, (_, row) in enumerate(cases.iterrows()):
        entry = traces_cache[row["Dataset"]]
        data_key = "gfp_data" if row["Channel"] == "GFP" else "mch_data"
        fit_key = "gfp_fit" if row["Channel"] == "GFP" else "mch_fit"
        plot_case(
            axes[0][col], axes[1][col],
            row["Dataset"], row["Channel"],
            entry["time_axis"], entry[data_key], entry[fit_key],
        )
    fig.suptitle("Worst (left) and best (right) single-exponential fits by runs-test p-value")
    fig.tight_layout()

    out_path = os.path.join(folder, "single_exponential_fit_check.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved diagnostic plot: {out_path}")


if __name__ == "__main__":
    main()
