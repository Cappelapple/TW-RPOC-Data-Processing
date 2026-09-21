"""Build matplotlib Figure objects from fit traces / pooled data.

These functions only ever return a Figure -- they never touch Tkinter.
The gui layer embeds the result via FigureCanvasTkAgg, or export code just
calls fig.savefig(...). That split is what lets "what a chart shows" change
independently of "how it's displayed."

Callers are responsible for closing figures they're done with (plt.close(fig))
and for calling gc.collect() afterward on the main/Tk thread -- see the
gc.disable() note in main.py for why that matters when the app is also
running a Cellpose background thread.
"""
import numpy as np
import matplotlib.pyplot as plt

from .fitting import exponential_decay

MARKER_BY_CONDITION = {"Normoxia": "o", "Hypoxia": "s", "Unknown": "^"}
LINESTYLES = ['-', '--', '-.', ':']
COLOR_BY_FLUOROPHORE_CONDITION = {
    ("GFP-LaminA", "Normoxia"): "#5CB85C", ("GFP-LaminA", "Hypoxia"): "#2E7D32",
    ("mCherry", "Normoxia"): "#D9534F", ("mCherry", "Hypoxia"): "#C0392B",
}


def _theme(dark_mode):
    return {
        "bg": "#2B2B2B" if dark_mode else "white",
        "face": "#1E1E1E" if dark_mode else "#F5F5F5",
        "text": "white" if dark_mode else "black",
        "grid": "gray" if dark_mode else "darkgray",
    }


def build_current_fit_figure(traces):
    """The two-panel (GFP / mCherry) fit-vs-data figure for one dataset's
    `traces` dict, as produced by intensity_traces.compute_corrected_traces
    plus the fitted (A, k, C) tuples. Always dark-themed (matches the
    in-app preview panel it's embedded in)."""
    time_axis = traces["time_axis"]
    gfp_t_corr = traces["gfp_data"]
    A_gfp, k_gfp, C_gfp = traces["gfp_fit"]
    mch_t_corr = traces["mch_data"]
    A_mch, k_mcherry, C_mch = traces["mch_fit"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4.5), facecolor="#2B2B2B")
    ax1.set_facecolor("#1E1E1E")
    ax2.set_facecolor("#1E1E1E")

    ax1.plot(time_axis, gfp_t_corr, 'go', alpha=0.4, label="Data")
    if not np.isnan(k_gfp):
        ax1.plot(time_axis, exponential_decay(time_axis, A_gfp, k_gfp, C_gfp), 'g-', label=f"k={k_gfp:.4f} frame⁻¹")
    if traces["has_gfp_control"]:
        c_lbl = "Control (Borrowed)" if traces.get("borrowed_control") else "Control (Bleach)"
        ax1.plot(time_axis, traces["gfp_control"], 'g--', label=c_lbl)
    ax1.set_title("GFP-Lamin A Decays", color="white", fontsize=10)
    ax1.set_xlabel("Frames", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(labelcolor="white", facecolor="#2B2B2B", edgecolor="none")

    ax2.plot(time_axis, mch_t_corr, 'ro', alpha=0.4, label="Data")
    if not np.isnan(k_mcherry):
        ax2.plot(time_axis, exponential_decay(time_axis, A_mch, k_mcherry, C_mch), 'r-', label=f"k={k_mcherry:.4f} frame⁻¹")
    if traces["has_mch_control"]:
        c_lbl = "Control (Borrowed)" if traces.get("borrowed_control") else "Control (Bleach)"
        ax2.plot(time_axis, traces["mch_control"], 'r--', label=c_lbl)
    ax2.set_title("mCherry Decays", color="white", fontsize=10)
    ax2.set_xlabel("Frames", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(labelcolor="white", facecolor="#2B2B2B", edgecolor="none")

    fig.tight_layout()
    return fig


def _power_groups(pool_df):
    powers = sorted(pool_df['Power_Num'].dropna().unique().tolist())
    return powers if powers else [None]


def _power_suffix(pwr):
    return f" @ {pwr:g}mW" if pwr is not None else ""


def _subset_by_power(pool_df, cond, pwr):
    if pwr is None:
        return pool_df[(pool_df["Condition"] == cond) & (pool_df['Power_Num'].isna())].sort_values('Wavelength_Num')
    return pool_df[(pool_df["Condition"] == cond) & (pool_df['Power_Num'] == pwr)].sort_values('Wavelength_Num')


def build_global_summary_figure(grouped_df, dark_mode=True):
    """Three-panel pooled response figure: GFP-only, mCherry-only, and a
    unified view, decay rate (k) vs wavelength, split by condition and
    (via linestyle) laser power. grouped_df comes from pooling.pool_replicates.
    Returns None if there's nothing to plot."""
    if grouped_df is None or grouped_df.empty:
        return None

    t = _theme(dark_mode)
    fig = plt.figure(figsize=(10, 7), facecolor=t["bg"])
    gfp_pool = grouped_df[grouped_df["Fluorophore"] == "GFP-LaminA"]
    mch_pool = grouped_df[grouped_df["Fluorophore"] == "mCherry"]

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.set_facecolor(t["face"])
    if not gfp_pool.empty:
        powers_present = _power_groups(gfp_pool)
        for cond, col in [("Normoxia", "#5CB85C"), ("Hypoxia", "#2E7D32")]:
            for p_idx, pwr in enumerate(powers_present):
                sub = _subset_by_power(gfp_pool, cond, pwr)
                if not sub.empty:
                    ls = LINESTYLES[p_idx % len(LINESTYLES)]
                    ax1.plot(sub['Wavelength_Num'], sub['Mean_k'], color=col, marker=MARKER_BY_CONDITION.get(cond, "o"), linestyle=ls, markersize=7, linewidth=2, label=f"GFP-LaminA {cond}{_power_suffix(pwr)}")
                    ax1.errorbar(sub['Wavelength_Num'], sub['Mean_k'], yerr=sub['Std_k'], fmt='none', ecolor=col, elinewidth=1.2, capsize=3)
    ax1.set_title("GFP-Lamin A Kinetics", color=t["text"], fontsize=11)
    ax1.tick_params(colors=t["text"])
    ax1.grid(True, linestyle=":", color=t["grid"], alpha=0.3)
    l1 = ax1.legend(facecolor=t["bg"], edgecolor="none", fontsize=8)
    if l1:
        for lbl in l1.get_texts():
            lbl.set_color(t["text"])

    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.set_facecolor(t["face"])
    if not mch_pool.empty:
        powers_present = _power_groups(mch_pool)
        for cond, col in [("Normoxia", "#D9534F"), ("Hypoxia", "#C0392B")]:
            for p_idx, pwr in enumerate(powers_present):
                sub = _subset_by_power(mch_pool, cond, pwr)
                if not sub.empty:
                    ls = LINESTYLES[p_idx % len(LINESTYLES)]
                    ax2.plot(sub['Wavelength_Num'], sub['Mean_k'], color=col, marker=MARKER_BY_CONDITION.get(cond, "o"), linestyle=ls, markersize=7, linewidth=2, label=f"mCherry {cond}{_power_suffix(pwr)}")
                    ax2.errorbar(sub['Wavelength_Num'], sub['Mean_k'], yerr=sub['Std_k'], fmt='none', ecolor=col, elinewidth=1.2, capsize=3)
    ax2.set_title("mCherry Kinetics", color=t["text"], fontsize=11)
    ax2.tick_params(colors=t["text"])
    ax2.grid(True, linestyle=":", color=t["grid"], alpha=0.3)
    l2 = ax2.legend(facecolor=t["bg"], edgecolor="none", fontsize=8)
    if l2:
        for lbl in l2.get_texts():
            lbl.set_color(t["text"])

    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    ax3.set_facecolor(t["face"])
    all_powers_present = _power_groups(grouped_df)
    for (fl, cond), grp_all in grouped_df.groupby(["Fluorophore", "Condition"]):
        c = COLOR_BY_FLUOROPHORE_CONDITION.get((fl, cond), "blue")
        for p_idx, pwr in enumerate(all_powers_present):
            if pwr is None:
                grp = grp_all[grp_all['Power_Num'].isna()].sort_values('Wavelength_Num')
            else:
                grp = grp_all[grp_all['Power_Num'] == pwr].sort_values('Wavelength_Num')
            if not grp.empty:
                ls = LINESTYLES[p_idx % len(LINESTYLES)]
                lbl = f"LaminA [{cond}]{_power_suffix(pwr)}" if "GFP-LaminA" in fl else f"mCherry [{cond}]{_power_suffix(pwr)}"
                ax3.plot(grp['Wavelength_Num'], grp['Mean_k'], color=c, marker=MARKER_BY_CONDITION.get(cond, "o"), linestyle=ls, markersize=7, linewidth=2, label=lbl)
                ax3.errorbar(grp['Wavelength_Num'], grp['Mean_k'], yerr=grp['Std_k'], fmt='none', ecolor=c, elinewidth=1.2, capsize=3)
    ax3.set_title("Unified Pooled Response Matrix", color=t["text"], fontsize=11)
    ax3.set_xlabel("Wavelength (nm)", color=t["text"])
    ax3.set_ylabel("Mean Decay Rate (k, frame⁻¹)", color=t["text"])
    ax3.tick_params(colors=t["text"])
    ax3.grid(True, linestyle=":", color=t["grid"], alpha=0.3)
    l3 = ax3.legend(facecolor=t["bg"], edgecolor="none", fontsize=8)
    if l3:
        for lbl in l3.get_texts():
            lbl.set_color(t["text"])

    fig.tight_layout()
    return fig


def build_power_response_figure(grouped_df, wavelength_filter, dark_mode=True):
    """Decay rate (k) vs laser power, at a fixed wavelength, split by
    fluorophore (panel) and condition (series). Returns None if there's no
    power-tagged data at that wavelength."""
    if grouped_df is None or grouped_df.empty:
        return None

    sub_df = grouped_df[
        np.isclose(grouped_df['Wavelength_Num'], wavelength_filter) & grouped_df['Power_Num'].notna()
    ]
    if sub_df.empty:
        return None

    t = _theme(dark_mode)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), facecolor=t["bg"])
    ax1.set_facecolor(t["face"])
    ax2.set_facecolor(t["face"])

    for fl, ax in [("GFP-LaminA", ax1), ("mCherry", ax2)]:
        fl_df = sub_df[sub_df["Fluorophore"] == fl]
        for cond, grp in fl_df.groupby("Condition"):
            grp = grp.sort_values('Power_Num')
            c = COLOR_BY_FLUOROPHORE_CONDITION.get((fl, cond), "steelblue")
            ax.plot(grp['Power_Num'], grp['Mean_k'], color=c, marker=MARKER_BY_CONDITION.get(cond, "o"), markersize=8, linewidth=2, label=cond)
            ax.errorbar(grp['Power_Num'], grp['Mean_k'], yerr=grp['Std_k'], fmt='none', ecolor=c, elinewidth=1.5, capsize=4)
        ax.set_title(f"{fl} — {wavelength_filter:g}nm", color=t["text"], fontsize=11)
        ax.set_xlabel("Laser Power (mW)", color=t["text"])
        ax.set_ylabel("Mean Decay Rate (k, frame⁻¹)", color=t["text"])
        ax.tick_params(colors=t["text"])
        ax.grid(True, linestyle=":", color=t["grid"], alpha=0.3)
        leg = ax.legend(facecolor=t["bg"], edgecolor="none")
        if leg:
            for lbl in leg.get_texts():
                lbl.set_color(t["text"])

    fig.suptitle(f"Power-Dependence of Decay Kinetics at {wavelength_filter:g}nm", color=t["text"], fontsize=12, weight="bold")
    fig.tight_layout()
    return fig


def _draw_spectral_panel(ax, pool_df, series, title, dark_mode):
    """Shared single-axis "Mean k vs Wavelength" plotting, used by both the
    per-fluorophore export figures below. series is a list of (condition,
    color) pairs, e.g. [("Normoxia", "#5CB85C"), ("Hypoxia", "#2E7D32")]."""
    t = _theme(dark_mode)
    ax.set_facecolor(t["face"])
    if not pool_df.empty:
        powers_present = _power_groups(pool_df)
        for cond, col in series:
            for p_idx, pwr in enumerate(powers_present):
                sub = _subset_by_power(pool_df, cond, pwr)
                if not sub.empty:
                    ls = LINESTYLES[p_idx % len(LINESTYLES)]
                    label = f"{title} {cond}{_power_suffix(pwr)}"
                    ax.plot(sub['Wavelength_Num'], sub['Mean_k'], color=col, marker=MARKER_BY_CONDITION.get(cond, "o"), linestyle=ls, markersize=8, linewidth=2, label=label)
                    ax.errorbar(sub['Wavelength_Num'], sub['Mean_k'], yerr=sub['Std_k'], fmt='none', ecolor=col, elinewidth=1.5, capsize=4)
    ax.set_title(f"Pooled {title} Action Decay Spectrum", color=t["text"])
    ax.set_xlabel("Wavelength (nm)", color=t["text"])
    ax.set_ylabel("Mean k (frame⁻¹)", color=t["text"])
    ax.tick_params(colors=t["text"])
    ax.grid(True, linestyle=":", color=t["grid"], alpha=0.3)
    leg = ax.legend(fontsize=8, facecolor=t["bg"], edgecolor="none")
    if leg:
        for lbl in leg.get_texts():
            lbl.set_color(t["text"])


def build_gfp_spectral_figure(grouped_df, dark_mode=False):
    """Standalone GFP-Lamin A-only spectral decay figure, for export."""
    t = _theme(dark_mode)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=t["bg"])
    gfp_pool = grouped_df[grouped_df["Fluorophore"] == "GFP-LaminA"]
    _draw_spectral_panel(ax, gfp_pool, [("Normoxia", "#5CB85C"), ("Hypoxia", "#2E7D32")], "GFP-Lamin A", dark_mode)
    fig.tight_layout()
    return fig


def build_mcherry_spectral_figure(grouped_df, dark_mode=False):
    """Standalone mCherry-only spectral decay figure, for export."""
    t = _theme(dark_mode)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=t["bg"])
    mch_pool = grouped_df[grouped_df["Fluorophore"] == "mCherry"]
    _draw_spectral_panel(ax, mch_pool, [("Normoxia", "#D9534F"), ("Hypoxia", "#C0392B")], "mCherry", dark_mode)
    fig.tight_layout()
    return fig


def build_combined_spectral_figure(grouped_df, dark_mode=False):
    """Standalone unified (both fluorophores) spectral decay figure, for export."""
    t = _theme(dark_mode)
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=t["bg"])
    ax.set_facecolor(t["face"])
    all_powers_present = _power_groups(grouped_df)
    for (fl, cond), grp_all in grouped_df.groupby(["Fluorophore", "Condition"]):
        c = COLOR_BY_FLUOROPHORE_CONDITION.get((fl, cond), "blue")
        for p_idx, pwr in enumerate(all_powers_present):
            if pwr is None:
                grp = grp_all[grp_all['Power_Num'].isna()].sort_values('Wavelength_Num')
            else:
                grp = grp_all[grp_all['Power_Num'] == pwr].sort_values('Wavelength_Num')
            if not grp.empty:
                ls = LINESTYLES[p_idx % len(LINESTYLES)]
                lbl = f"LaminA [{cond}]{_power_suffix(pwr)}" if "GFP-LaminA" in fl else f"mCherry [{cond}]{_power_suffix(pwr)}"
                ax.plot(grp['Wavelength_Num'], grp['Mean_k'], color=c, marker=MARKER_BY_CONDITION.get(cond, "o"), linestyle=ls, markersize=8, linewidth=2, label=lbl)
                ax.errorbar(grp['Wavelength_Num'], grp['Mean_k'], yerr=grp['Std_k'], fmt='none', ecolor=c, elinewidth=1.5, capsize=4)
    ax.set_title("Unified Action Spectroscopy Decay Spectrum", color=t["text"])
    ax.set_xlabel("Wavelength (nm)", color=t["text"])
    ax.set_ylabel("Mean Decay Rate (k, frame⁻¹)", color=t["text"])
    ax.tick_params(colors=t["text"])
    ax.grid(True, linestyle=":", color=t["grid"], alpha=0.3)
    leg = ax.legend(fontsize=8, facecolor=t["bg"], edgecolor="none")
    if leg:
        for lbl in leg.get_texts():
            lbl.set_color(t["text"])
    fig.tight_layout()
    return fig


def build_individual_decay_figure(dataset_name, traces):
    """Per-dataset publication export figure (light-themed), for
    export_publication_plots. Distinct from build_current_fit_figure, which
    is dark-themed for the in-app preview panel."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
    t_axis = traces["time_axis"]

    ax1.plot(t_axis, traces["gfp_data"], 'go', alpha=0.5, label="Data")
    A, k, C = traces["gfp_fit"]
    if not np.isnan(k):
        ax1.plot(t_axis, exponential_decay(t_axis, A, k, C), 'g-', label=f"Fit (k={k:.4f} frame⁻¹)")
    if traces["has_gfp_control"]:
        c_lbl = "Control (Borrowed)" if traces.get("borrowed_control") else "Control (Bleach)"
        ax1.plot(t_axis, traces["gfp_control"], 'g--', label=c_lbl)
    ax1.set_title("GFP-Lamin A Profile")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("F/F0 (Corrected)")
    ax1.grid(True, linestyle=":")
    ax1.legend()

    ax2.plot(t_axis, traces["mch_data"], 'ro', alpha=0.5, label="Data")
    A_m, k_m, C_m = traces["mch_fit"]
    if not np.isnan(k_m):
        ax2.plot(t_axis, exponential_decay(t_axis, A_m, k_m, C_m), 'r-', label=f"Fit (k={k_m:.4f} frame⁻¹)")
    if traces["has_mch_control"]:
        c_lbl = "Control (Borrowed)" if traces.get("borrowed_control") else "Control (Bleach)"
        ax2.plot(t_axis, traces["mch_control"], 'r--', label=c_lbl)
    ax2.set_title("mCherry Profile")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("F/F0 (Corrected)")
    ax2.grid(True, linestyle=":")
    ax2.legend()

    fig.suptitle(f"Decay Profile: {dataset_name}", fontsize=11, weight="bold")
    fig.tight_layout()
    return fig
