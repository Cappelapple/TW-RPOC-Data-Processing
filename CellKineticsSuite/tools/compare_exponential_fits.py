"""Single- vs two-exponential fit comparison.

Both fits are on the ratio-corrected trace (control already divided out).
The two-exponential fit is fully free -- no rate is borrowed/fixed from the
control's own fit. See core/fitting.py's fit_biexponential_decay docstring
for why.

Usage:
    py -3.12 tools/compare_exponential_fits.py "<results folder>"
        Lists cached dataset names. Doesn't fit anything.

    py -3.12 tools/compare_exponential_fits.py "<results folder>" <dataset_name> [<dataset_name> ...]
        Prints the comparison for just those datasets and saves one
        data+fit+residuals plot per dataset/channel.

    py -3.12 tools/compare_exponential_fits.py "<results folder>" --all
        Runs every cached dataset, writes one summary CSV
        (exponential_fit_comparison.csv) to the folder, and saves one
        combined plot of the cases AIC favors two-exponential most strongly.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cell_kinetics.core import diagnostics, fitting, persistence  # noqa: E402


def compute_comparison(time_axis, data):
    """Fit both models to one trace. Returns a dict with every number and
    array a caller might want -- printing, CSV rows, and plotting all pull
    from this so there's one place the fitting actually happens."""
    single_params = fitting.fit_decay_constant(time_axis, data)
    bi_params = fitting.fit_biexponential_decay(time_axis, data)

    r_single = diagnostics.fit_residuals(time_axis, data, single_params)
    model_single = np.asarray(data, dtype=float) - r_single

    if np.any(np.isnan(bi_params)):
        model_bi = np.full_like(np.asarray(data, dtype=float), np.nan)
        r_bi = model_bi.copy()
    else:
        model_bi = fitting.biexponential_decay(np.asarray(time_axis, dtype=float), *bi_params)
        r_bi = np.asarray(data, dtype=float) - model_bi

    aic_single = diagnostics.aic(r_single, n_params=3)
    aic_bi = diagnostics.aic(r_bi, n_params=5)
    bic_single = diagnostics.bic(r_single, n_params=3)
    bic_bi = diagnostics.bic(r_bi, n_params=5)
    delta_aic = aic_single - aic_bi  # positive => two-exponential preferred

    return {
        "single_params": single_params,
        "bi_params": bi_params,
        "model_single": model_single,
        "model_bi": model_bi,
        "r_single": r_single,
        "r_bi": r_bi,
        "aic_single": aic_single,
        "aic_bi": aic_bi,
        "bic_single": bic_single,
        "bic_bi": bic_bi,
        "delta_aic": delta_aic,
        "r2_single": diagnostics.r_squared(data, r_single),
        "r2_bi": diagnostics.r_squared(data, r_bi),
    }


def print_comparison(label, result):
    print(f"\n=== {label} ===")
    A, k, C = result["single_params"]
    print(f"  single-exponential: A={A:.4f} k={k:.4f} C={C:.4f}"
          f"  | AIC={result['aic_single']:.2f} BIC={result['bic_single']:.2f}"
          f"  | R2={result['r2_single']:.4f}")

    if np.any(np.isnan(result["bi_params"])):
        print("  two-exponential:    fit did not converge")
        return
    A1, k1, A2, k2, C2 = result["bi_params"]
    print(f"  two-exponential:    A1={A1:.4f} k1={k1:.4f} (fast)  A2={A2:.4f} k2={k2:.4f} (slow)  C={C2:.4f}"
          f"  | AIC={result['aic_bi']:.2f} BIC={result['bic_bi']:.2f}"
          f"  | R2={result['r2_bi']:.4f}")
    winner = "two-exponential" if result["delta_aic"] > 0 else "single-exponential"
    print(f"  -> AIC prefers {winner} (delta={abs(result['delta_aic']):.2f}; "
          f">2 is usually considered meaningful, >10 strong)")


def plot_comparison(time_axis, data, result, label, out_path):
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax_top.plot(time_axis, data, "o", ms=4, color="#2c3e50", label="data")
    ax_top.plot(time_axis, result["model_single"], "-", color="#e74c3c", label="single-exp fit")
    if not np.any(np.isnan(result["model_bi"])):
        ax_top.plot(time_axis, result["model_bi"], "-", color="#27ae60", label="two-exp fit")
    ax_top.set_title(label)
    ax_top.legend(fontsize=8)

    ax_bottom.axhline(0, color="#7f8c8d", lw=1)
    ax_bottom.plot(time_axis, result["r_single"], "o", ms=4, color="#e74c3c", alpha=0.7, label="single-exp residuals")
    if not np.any(np.isnan(result["r_bi"])):
        ax_bottom.plot(time_axis, result["r_bi"], "o", ms=4, color="#27ae60", alpha=0.7, label="two-exp residuals")
    ax_bottom.set_xlabel("frame")
    ax_bottom.set_ylabel("residual")
    ax_bottom.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_named(folder, traces_cache, requested):
    missing = [name for name in requested if name not in traces_cache]
    if missing:
        print(f"Not found in the cache: {missing}")
        sys.exit(1)

    for name in requested:
        entry = traces_cache[name]
        time_axis = entry["time_axis"]
        for channel, data_key in (("GFP", "gfp_data"), ("mCherry", "mch_data")):
            data = entry.get(data_key)
            if data is None:
                continue
            label = f"{name} ({channel})"
            result = compute_comparison(time_axis, data)
            print_comparison(label, result)

            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
            out_path = os.path.join(folder, f"{safe_name}_{channel}_fit_comparison.png")
            plot_comparison(time_axis, data, result, label, out_path)
            print(f"  saved: {out_path}")


def run_all(folder, traces_cache):
    rows = []
    results_by_key = {}

    for name, entry in traces_cache.items():
        time_axis = entry["time_axis"]
        for channel, data_key in (("GFP", "gfp_data"), ("mCherry", "mch_data")):
            data = entry.get(data_key)
            if data is None:
                continue
            result = compute_comparison(time_axis, data)
            results_by_key[(name, channel)] = (time_axis, data, result)

            A, k, C = result["single_params"]
            bi = result["bi_params"]
            rows.append({
                "Dataset": name,
                "Channel": channel,
                "k_single": k,
                "AIC_single": result["aic_single"],
                "R2_single": result["r2_single"],
                "A1_fast": bi[0], "k1_fast": bi[1], "A2_slow": bi[2], "k2_slow": bi[3], "C_bi": bi[4],
                "AIC_bi": result["aic_bi"],
                "R2_bi": result["r2_bi"],
                "delta_AIC": result["delta_aic"],
                "prefers_two_exp": bool(result["delta_aic"] > 2.0),
            })

    if not rows:
        print("Cache had entries but none had a trace to fit.")
        return

    df = pd.DataFrame(rows).sort_values("delta_AIC", ascending=False, na_position="last").reset_index(drop=True)
    csv_path = os.path.join(folder, "exponential_fit_comparison.csv")
    df.to_csv(csv_path, index=False)

    n_prefers_two = int(df["prefers_two_exp"].sum())
    n_total = int(df["delta_AIC"].notna().sum())
    print(f"Wrote {csv_path}")
    print(f"{n_prefers_two}/{n_total} dataset/channel fits prefer two-exponential at AIC delta>2")

    top = df.dropna(subset=["delta_AIC"]).head(6)
    if top.empty:
        return

    fig, axes = plt.subplots(2, len(top), figsize=(3.2 * len(top), 5), squeeze=False)
    for col, (_, row) in enumerate(top.iterrows()):
        time_axis, data, result = results_by_key[(row["Dataset"], row["Channel"])]
        ax_top, ax_bottom = axes[0][col], axes[1][col]

        ax_top.plot(time_axis, data, "o", ms=3, color="#2c3e50")
        ax_top.plot(time_axis, result["model_single"], "-", color="#e74c3c", lw=1)
        if not np.any(np.isnan(result["model_bi"])):
            ax_top.plot(time_axis, result["model_bi"], "-", color="#27ae60", lw=1)
        ax_top.set_title(f"{row['Dataset']} ({row['Channel']})\ndelta_AIC={row['delta_AIC']:.1f}", fontsize=8)

        ax_bottom.axhline(0, color="#7f8c8d", lw=1)
        ax_bottom.plot(time_axis, result["r_single"], "o", ms=3, color="#e74c3c")
        if not np.any(np.isnan(result["r_bi"])):
            ax_bottom.plot(time_axis, result["r_bi"], "o", ms=3, color="#27ae60")

    fig.suptitle("Strongest cases for two-exponential over single-exponential (by AIC)")
    fig.tight_layout()
    plot_path = os.path.join(folder, "exponential_fit_comparison_top_cases.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote {plot_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: py -3.12 tools/compare_exponential_fits.py \"<results folder>\" [dataset_name ... | --all]")
        sys.exit(1)

    folder = sys.argv[1]
    requested = sys.argv[2:]

    traces_cache, _ = persistence.load_session_cache(folder)
    if not traces_cache:
        print(f"No {persistence.CACHE_FILENAME} found (or it's empty) in: {folder}")
        sys.exit(1)

    if requested == ["--all"]:
        run_all(folder, traces_cache)
        return

    if not requested:
        print("No dataset name given. Available datasets:")
        for name in sorted(traces_cache):
            print(f"  {name}")
        print("\nRe-run with one or more of these names, or with --all to run every dataset and save a summary CSV.")
        sys.exit(0)

    run_named(folder, traces_cache, requested)


if __name__ == "__main__":
    main()
