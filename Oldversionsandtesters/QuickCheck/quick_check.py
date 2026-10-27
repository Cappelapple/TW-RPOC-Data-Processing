"""Bare-bones photobleaching validity check -- no GUI, no cell segmentation.

Point this at a working folder and, for every dataset it finds, it plots two
curves per channel: the average intensity inside the laser-target mask
("Treated") and the average intensity of everything else in the frame
("Rest"), each normalized to its own first frame (F/F0). That's enough to
eyeball in seconds whether a run actually bleached where it was supposed to
-- Treated should visibly drop, Rest should stay roughly flat -- without
waiting for cell segmentation or running the full analysis pipeline.

This is a go/no-go check while you're still at the microscope, not the real
analysis. For per-cell classification, control-division, curve fitting, and
everything else, use CellKineticsSuite.

It reuses CellKineticsSuite's file-matching and mask-threshold logic (both
plain numpy, no Cellpose/torch/customtkinter needed) rather than
re-implementing them -- so it lives in this repo as a sibling folder to
CellKineticsSuite, not as a fully standalone tool.

Usage:
    py -3.12 quick_check.py "<folder>" <x_pixels> <y_pixels>

x_pixels/y_pixels are the two values every flat-text channel file needs to
be reshaped correctly. They're asked for explicitly, rather than read from
a Parameters.txt, because this is meant to work on a still-growing file: it
uses however many complete frames exist in the GFP/mCherry files right now,
so you can check a long acquisition partway through instead of waiting for
it to finish -- and pixel dimensions are fixed from the moment a scan
starts, unlike frame count.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CellKineticsSuite"))

from cell_kinetics.core import dataset_io, segmentation  # noqa: E402


def normalized_region_trace(stack, region_mask):
    """Per-frame mean intensity inside region_mask, normalized to frame 0.
    NaN trace if the mask is empty or the first frame's mean is ~0."""
    if not np.any(region_mask):
        return np.full(stack.shape[0], np.nan)
    raw = stack[:, region_mask].mean(axis=1)
    if abs(raw[0]) < 1e-9:
        return np.full_like(raw, np.nan)
    return raw / raw[0]


def load_partial_stack(path, x_pixels, y_pixels):
    """Like dataset_io.load_dataset_stacks, but for one file that might
    still be growing mid-acquisition -- uses however many complete frames
    are present right now rather than requiring a frame count up front.

    Returns (stack, num_frames). Raises ValueError if there isn't yet even
    one complete frame at the given resolution.
    """
    data = dataset_io.parse_data_file(path)
    pts_per_frame = x_pixels * y_pixels
    num_frames = len(data) // pts_per_frame
    if num_frames < 1:
        raise ValueError(f"{os.path.basename(path)} has fewer than one complete frame yet.")
    return data[: num_frames * pts_per_frame].reshape((num_frames, y_pixels, x_pixels)), num_frames


def check_one_dataset(folder, prefix, all_files, x_pixels, y_pixels):
    channel_files = dataset_io.find_channel_files(prefix, all_files)
    if not channel_files["mask"]:
        print(f"[{prefix}] no mask file yet -- skipping (need it to define the treated region)")
        return

    mask_stack, _ = load_partial_stack(os.path.join(folder, channel_files["mask"]), x_pixels, y_pixels)
    treated_mask = segmentation.build_laser_mask(mask_stack[0], mask_stack[0].shape)
    rest_mask = ~treated_mask

    if not np.any(treated_mask):
        print(f"[{prefix}] mask is empty -- nothing marked as treated")
        return

    channels = []
    for label, key in (("GFP", "gfp"), ("mCherry", "mcherry")):
        fname = channel_files[key]
        if not fname:
            continue
        try:
            stack, n_frames = load_partial_stack(os.path.join(folder, fname), x_pixels, y_pixels)
        except ValueError as e:
            print(f"[{prefix}] {label}: {e}")
            continue
        treated_trace = normalized_region_trace(stack, treated_mask)
        rest_trace = normalized_region_trace(stack, rest_mask)
        channels.append((label, n_frames, treated_trace, rest_trace))

    if not channels:
        print(f"[{prefix}] no GFP/mCherry data yet")
        return

    note = dataset_io.dataset_note(channel_files)
    note_str = f"  (⚠ annotated: {note})" if note else ""
    print(f"\n[{prefix}]{note_str}  Treated: {int(np.sum(treated_mask))}px  |  Rest: {int(np.sum(rest_mask))}px")

    for label, n_frames, treated_trace, rest_trace in channels:
        treated_drop = 1.0 - treated_trace[-1] if not np.isnan(treated_trace[-1]) else float("nan")
        rest_drop = 1.0 - rest_trace[-1] if not np.isnan(rest_trace[-1]) else float("nan")
        looks_valid = treated_drop > 0.05 and treated_drop > 2 * max(rest_drop, 1e-6)
        verdict = "looks valid" if looks_valid else "check this one"
        print(f"  {label} ({n_frames} frames so far): Treated dropped {treated_drop * 100:5.1f}%"
              f"  |  Rest dropped {rest_drop * 100:5.1f}%  ->  {verdict}")

    fig, axes = plt.subplots(1, len(channels), figsize=(5.5 * len(channels), 4), squeeze=False)
    for col, (label, n_frames, treated_trace, rest_trace) in enumerate(channels):
        ax = axes[0][col]
        t = np.arange(n_frames)
        ax.plot(t, treated_trace, "r-o", ms=3, label="Treated (mask)")
        ax.plot(t, rest_trace, "b-o", ms=3, label="Rest of frame")
        ax.axhline(1.0, color="gray", lw=0.5)
        ax.set_title(f"{prefix} — {label}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("F / F0")
        ax.legend()
    fig.tight_layout()

    out_path = os.path.join(folder, f"{prefix}_quick_check.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved: {out_path}")


def check_folder(folder, x_pixels, y_pixels):
    all_files = os.listdir(folder)
    prefixes = sorted(set(dataset_io.split_prefix(f) for f in all_files if f.lower().endswith(".txt")))
    if not prefixes:
        print(f"No .txt files found in {folder}")
        return
    for prefix in prefixes:
        try:
            check_one_dataset(folder, prefix, all_files, x_pixels, y_pixels)
        except Exception as e:
            print(f"[{prefix}] error: {e}")


def main():
    # Windows consoles often default to a codepage (e.g. cp1252) that can't
    # encode characters like the warning triangle used below -- without
    # this, a single annotated-file note crashes that dataset's whole
    # report instead of just printing.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if len(sys.argv) != 4:
        print('Usage: py -3.12 quick_check.py "<folder>" <x_pixels> <y_pixels>')
        sys.exit(1)
    folder = sys.argv[1]
    x_pixels = int(sys.argv[2])
    y_pixels = int(sys.argv[3])
    check_folder(folder, x_pixels, y_pixels)


if __name__ == "__main__":
    main()
