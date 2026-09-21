"""Discover and load flat-text dataset files.

A "dataset" is a prefix (e.g. "1000nm_40mW_normoxia_v3") with 3-4 files
alongside it: a GFP/LaminA channel stack, an mCherry/Chromatin channel
stack, a single-frame laser-target Mask, and an optional Parameters file.

This module is the single source of truth for that file-matching logic —
previously it was duplicated almost verbatim between the local-folder loader
and the Box background-sync worker, which is exactly the kind of drift that
causes one path to work and the other to silently miss files.
"""
import os
import re

import numpy as np

_CHANNEL_SPLIT_RE = re.compile(r'_(GFP|mCherry|LaminA|Chromatin|Mask|Parameter|Parameters)', re.IGNORECASE)


def split_prefix(filename):
    """'1000nm_40mW_normoxia_v3_GFP.txt' -> '1000nm_40mW_normoxia_v3'"""
    return _CHANNEL_SPLIT_RE.split(filename)[0]


def find_channel_files(prefix, file_list):
    """Find this prefix's GFP/mCherry/Mask/Parameters files (case-insensitive)
    within file_list. Any entry not found is None -- callers decide whether
    that's fatal (a fully local folder scan) or just "still pending" (Box sync).

    Returns a dict: {"gfp": ..., "mcherry": ..., "mask": ..., "param": ...}
    """
    lower_prefix = prefix.lower()
    lower_map = {f.lower(): f for f in file_list}

    def first_of(*candidates):
        for c in candidates:
            if c in lower_map:
                return lower_map[c]
        return None

    return {
        "gfp": first_of(f"{lower_prefix}_gfp.txt", f"{lower_prefix}_lamina.txt"),
        "mcherry": first_of(f"{lower_prefix}_mcherry.txt", f"{lower_prefix}_chromatin.txt"),
        "mask": first_of(f"{lower_prefix}_mask.txt"),
        "param": first_of(f"{lower_prefix}_parameters.txt", f"{lower_prefix}_parameter.txt"),
    }


def is_complete(channel_files):
    """True once gfp/mcherry/mask are all present (param file is optional)."""
    return bool(channel_files["gfp"] and channel_files["mcherry"] and channel_files["mask"])


def missing_channels(channel_files):
    """Human-readable list of which required channels are missing, e.g.
    ["GFP/LaminA", "Mask"]. Empty list means complete."""
    missing = []
    if not channel_files["gfp"]:
        missing.append("GFP/LaminA")
    if not channel_files["mcherry"]:
        missing.append("mCherry/Chromatin")
    if not channel_files["mask"]:
        missing.append("Mask")
    return missing


def discover_dataset_prefixes(file_list):
    """All .txt-file prefixes in file_list that have a complete GFP+mCherry+Mask
    triplet, sorted. This is the "ready to load" set for a local folder scan."""
    prefixes = set()
    for f in file_list:
        if f.lower().endswith(".txt"):
            prefixes.add(split_prefix(f))

    valid = [p for p in prefixes if is_complete(find_channel_files(p, file_list))]
    return sorted(valid)


def parse_data_file(path):
    """Read a flat-text intensity file into a 1D float array."""
    with open(path, 'r') as f:
        content = f.read()
    return np.array(re.findall(r'[-+]?\d*\.\d+|\d+', content), dtype=np.float64)


def load_dataset_stacks(gfp_path, mcherry_path, mask_path, x_pixels, y_pixels):
    """Load and reshape the GFP/mCherry stacks and the single-frame mask.

    Returns (gfp_stack, mcherry_stack, mask_frame, num_frames).
    Raises ValueError if there aren't enough points for even one frame at
    the given resolution.
    """
    gfp_data = parse_data_file(gfp_path)
    mcherry_data = parse_data_file(mcherry_path)
    mask_data = parse_data_file(mask_path)

    pts_per_frame = x_pixels * y_pixels
    num_frames = len(gfp_data) // pts_per_frame
    if num_frames < 1:
        raise ValueError(
            f"File {os.path.basename(gfp_path)} has insufficient points "
            f"({len(gfp_data)}) for {x_pixels}x{y_pixels} resolution."
        )

    total_pts = num_frames * pts_per_frame
    gfp_stack = gfp_data[:total_pts].reshape((num_frames, y_pixels, x_pixels))
    mcherry_stack = mcherry_data[:total_pts].reshape((num_frames, y_pixels, x_pixels))
    mask_frame = mask_data[:pts_per_frame].reshape((y_pixels, x_pixels))

    return gfp_stack, mcherry_stack, mask_frame, num_frames
