"""Bare-bones, no-UI reproducibility check across multiple days of data.

Combines datasets from several folders (one per imaging day, typically),
computes a decay constant per trial, pools replicates by
wavelength/power/condition/fluorophore across ALL folders combined, and
flags + excludes whichever replicate(s) in each group would shrink that
group's spread the most -- so you can see how reproducible your results
are day to day, with the worst outliers already pulled out instead of
quietly inflating the error bars.

A real simplification worth knowing about: like QuickCheck, this skips
per-cell segmentation entirely. "Treated" is the laser-target mask region;
"control" is literally everything else in the frame (real control cells and
background mixed together), not classified control cells. The division
step that removes imaging-induced bleaching is the same math
CellKineticsSuite's real pipeline uses (Treated_norm / Rest_norm), just
applied to a coarser region. Treat this as a fast reproducibility sanity
check across days, not a replacement for the real per-cell analysis.

This file is fully separate from QuickCheck and from CellKineticsSuite --
it does not import quick_check.py, and only *reads* CellKineticsSuite's
core modules (file-matching, mask-threshold, fitting, pooling, plotting --
all plain numpy/pandas/matplotlib, no Cellpose/torch/customtkinter) to
reuse already-tested logic. Nothing in either of those is modified.

Usage:
    py -3.12 reproducibility_check.py <x_pixels> <y_pixels> <day1_folder> [<day2_folder> ...]

Output, written to the first folder given:
    reproducibility_summary.csv   one row per dataset/channel found, with
                                   Excluded/ExcludeReason columns
    reproducibility_report.png    pooled k vs wavelength (post-exclusion),
                                   error bars from Std_k
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CellKineticsSuite"))

from cell_kinetics.core import dataset_io, fitting, metadata_parsing, plotting, pooling, segmentation  # noqa: E402


def load_full_stack(path, x_pixels, y_pixels):
    """Reshape a flat-text channel file using however many complete frames
    it actually contains. Returns (stack, num_frames); raises ValueError if
    there isn't even one complete frame at the given resolution."""
    data = dataset_io.parse_data_file(path)
    pts_per_frame = x_pixels * y_pixels
    num_frames = len(data) // pts_per_frame
    if num_frames < 1:
        raise ValueError(f"{os.path.basename(path)} has fewer than one complete frame.")
    return data[: num_frames * pts_per_frame].reshape((num_frames, y_pixels, x_pixels)), num_frames


def corrected_trace(stack, treated_mask, rest_mask):
    """Treated/Rest, each normalized to its own frame 0 first -- the same
    division CellKineticsSuite's control-correction uses, applied to the
    mask region vs. everything else instead of classified cells. NaN trace
    if either region's first frame is ~0."""
    treated_raw = stack[:, treated_mask].mean(axis=1)
    rest_raw = stack[:, rest_mask].mean(axis=1)
    if abs(treated_raw[0]) < 1e-9 or abs(rest_raw[0]) < 1e-9:
        return np.full(stack.shape[0], np.nan)
    treated_norm = treated_raw / treated_raw[0]
    rest_norm = rest_raw / rest_raw[0]
    return treated_norm / np.clip(rest_norm, 1e-4, None)


def process_dataset(folder, prefix, all_files):
    """One dataset -> list of result-row dicts (one per GFP/mCherry file
    present), or [] if there's no usable mask. x_pixels/y_pixels are
    closed over by the caller via functools.partial-style wiring below."""
    channel_files = dataset_io.find_channel_files(prefix, all_files)
    if not channel_files["mask"]:
        print(f"  [{prefix}] no mask file -- skipping")
        return []
    return channel_files


def analyze_dataset(folder, prefix, channel_files, x_pixels, y_pixels):
    try:
        mask_stack, _ = load_full_stack(os.path.join(folder, channel_files["mask"]), x_pixels, y_pixels)
    except ValueError as e:
        print(f"  [{prefix}] mask: {e}")
        return []
    treated_mask = segmentation.build_laser_mask(mask_stack[0], mask_stack[0].shape)
    rest_mask = ~treated_mask
    if not np.any(treated_mask) or not np.any(rest_mask):
        print(f"  [{prefix}] mask is empty or covers the whole frame -- skipping")
        return []

    power_from_param = None
    if channel_files["param"]:
        try:
            with open(os.path.join(folder, channel_files["param"]), "r") as f:
                params = metadata_parsing.parse_parameters_file(f.read())
            power_from_param = params.get("power")
        except Exception:
            pass
    power = metadata_parsing.detect_power_from_text(prefix) or power_from_param or "Unknown"

    wave = metadata_parsing.detect_wavelength_from_text(prefix)
    wavelength = f"{wave}nm" if wave else "Unknown"
    condition = metadata_parsing.detect_condition_from_text(prefix)
    note = dataset_io.dataset_note(channel_files)

    rows = []
    for fluorophore, key in (("GFP-LaminA", "gfp"), ("mCherry", "mcherry")):
        fname = channel_files[key]
        if not fname:
            continue
        try:
            stack, num_frames = load_full_stack(os.path.join(folder, fname), x_pixels, y_pixels)
        except ValueError as e:
            print(f"  [{prefix}] {fluorophore}: {e}")
            continue

        corrected = corrected_trace(stack, treated_mask, rest_mask)
        time_axis = np.arange(num_frames)
        _, k, _ = fitting.fit_decay_constant(time_axis, corrected)

        rows.append({
            "Dataset": prefix, "SourceFolder": folder, "Wavelength": wavelength,
            "Condition": condition, "Power_mW": power, "Fluorophore": fluorophore,
            "Decay_Constant_k": k, "N_Frames": num_frames,
            "Excluded": bool(note), "ExcludeReason": f"annotated: {note}" if note else "",
        })
    return rows


def collect_folders(folders, x_pixels, y_pixels):
    rows = []
    for folder in folders:
        all_files = os.listdir(folder)
        prefixes = sorted(set(dataset_io.split_prefix(f) for f in all_files if f.lower().endswith(".txt")))
        print(f"\n{folder}: {len(prefixes)} dataset(s) found")
        for prefix in prefixes:
            channel_files = process_dataset(folder, prefix, all_files)
            if not channel_files:
                continue
            rows.extend(analyze_dataset(folder, prefix, channel_files, x_pixels, y_pixels))
    return pd.DataFrame(rows)


def flag_and_exclude_outliers(df):
    """One leave-one-out pass per Wavelength/Power/Condition/Fluorophore
    group, among replicates not already excluded (e.g. annotated files):
    flags and excludes whichever replicate(s) would shrink that group's
    spread the most. Same diagnostic CellKineticsSuite's Manage Data window
    uses interactively, applied automatically since there's no UI here to
    click a checkbox."""
    df = df.copy()
    df["Wavelength_Num"] = df["Wavelength"].astype(str).str.extract(r"(\d+)").astype(float)

    group_cols = ["Wavelength_Num", "Power_mW", "Condition", "Fluorophore"]
    for _, group_df in df.groupby(group_cols, dropna=False):
        candidates = group_df[(~group_df["Excluded"]) & group_df["Decay_Constant_k"].notna()]
        if len(candidates) < 2:
            continue
        flags = pooling.leave_one_out_outlier_flags(dict(zip(candidates.index, candidates["Decay_Constant_k"])))
        for idx, (current_std, new_std) in flags.items():
            df.loc[idx, "Excluded"] = True
            df.loc[idx, "ExcludeReason"] = f"outlier (sigma {current_std:.4f} -> {new_std:.4f} if removed)"
            row = df.loc[idx]
            print(f"  excluded: {row['Dataset']} [{row['SourceFolder']}]  {row['Fluorophore']}  "
                  f"{row['Wavelength']} {row['Power_mW']}mW {row['Condition']}  "
                  f"k={row['Decay_Constant_k']:.4f}  sigma {current_std:.4f} -> {new_std:.4f}")
    return df


def print_group(label, grouped):
    print(f"\n--- {label} ---")
    if grouped is None or grouped.empty:
        print("  (no groups)")
        return
    for _, row in grouped.sort_values(["Fluorophore", "Wavelength_Num"]).iterrows():
        pwr = f"{row['Power_Num']:g}mW" if pd.notna(row["Power_Num"]) else "power unknown"
        print(f"  {row['Fluorophore']:11s} {row['Wavelength_Num']:5.0f}nm  {pwr:9s} {row['Condition']:9s}"
              f"  k = {row['Mean_k']:.4f} ± {row['Std_k']:.4f}  (N={row['Count']})")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if len(sys.argv) < 4:
        print("Usage: py -3.12 reproducibility_check.py <x_pixels> <y_pixels> <day1_folder> [<day2_folder> ...]")
        sys.exit(1)

    x_pixels, y_pixels = int(sys.argv[1]), int(sys.argv[2])
    folders = [os.path.abspath(f) for f in sys.argv[3:]]
    for f in folders:
        if not os.path.isdir(f):
            print(f"Not a folder: {f}")
            sys.exit(1)

    print(f"Combining {len(folders)} folder(s)...")
    df = collect_folders(folders, x_pixels, y_pixels)
    if df.empty:
        print("\nNo usable datasets found.")
        sys.exit(1)
    print(f"\nCollected {len(df)} dataset/channel result(s) from {df['Dataset'].nunique()} unique dataset name(s) "
          f"across {len(folders)} folder(s).")

    n_annotated = int(df["Excluded"].sum())
    if n_annotated:
        print(f"{n_annotated} pre-excluded as lab-annotated (see ExcludeReason).")

    print("\nFlagging statistical outliers (leave-one-out, per group)...")
    df = flag_and_exclude_outliers(df)
    n_excluded = int(df["Excluded"].sum())
    print(f"{n_excluded} of {len(df)} excluded in total.")

    before_df = df.copy()
    before_df["Excluded"] = False
    grouped_before, _ = pooling.pool_replicates(before_df)
    grouped_after, _ = pooling.pool_replicates(df)
    print_group("BEFORE exclusion", grouped_before)
    print_group("AFTER exclusion", grouped_after)

    out_folder = folders[0]
    csv_path = os.path.join(out_folder, "reproducibility_summary.csv")
    df.drop(columns=["Wavelength_Num"], errors="ignore").to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    fig = plotting.build_global_summary_figure(grouped_after, dark_mode=False)
    if fig is not None:
        png_path = os.path.join(out_folder, "reproducibility_report.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {png_path}")
    else:
        print("Nothing left to plot after exclusion.")


if __name__ == "__main__":
    main()
