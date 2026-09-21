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


_CHANNEL_ALTS = {
    "gfp": ["gfp", "lamina"],
    "mcherry": ["mcherry", "chromatin"],
    "mask": ["mask"],
    "param": ["parameters", "parameter"],
}


def _fuzzy_channel_match(prefix, channel_alts, file_list):
    """Find a file named '{prefix}_<channel><anything>.txt' where <channel>
    is one of channel_alts -- e.g. a lab-added quality note like
    '..._mCherry_treated weak one.txt'. Only called after an exact-suffix
    lookup has already failed, so a match here always carries a non-empty
    note (an exact match would have been found by the fast path already).

    Returns (filename, note) or (None, None).
    """
    alt_pattern = "|".join(re.escape(a) for a in channel_alts)
    pattern = re.compile(rf'^{re.escape(prefix)}_(?:{alt_pattern})(.*)\.txt$', re.IGNORECASE)
    for f in file_list:
        m = pattern.match(f)
        if m:
            note = m.group(1).strip(' _-')
            return f, (note or None)
    return None, None


def find_channel_files(prefix, file_list):
    """Find this prefix's GFP/mCherry/Mask/Parameters files (case-insensitive)
    within file_list. Any entry not found is None -- callers decide whether
    that's fatal (a fully local folder scan) or just "still pending" (Box sync).

    Falls back to a fuzzy match (prefix + channel + arbitrary trailing text)
    when the exact filename isn't found, so a lab-annotated file like
    '..._mCherry_treated weak one.txt' is still discovered instead of
    silently making the dataset look incomplete. Any such match is recorded
    in the "notes" dict so callers can flag it instead of treating it as a
    normal, unremarkable file.

    Returns a dict: {"gfp": ..., "mcherry": ..., "mask": ..., "param": ...,
    "notes": {"gfp": ..., "mcherry": ..., "mask": ..., "param": ...}}
    """
    lower_prefix = prefix.lower()
    lower_map = {f.lower(): f for f in file_list}

    def first_of(*candidates):
        for c in candidates:
            if c in lower_map:
                return lower_map[c]
        return None

    result = {
        "gfp": first_of(f"{lower_prefix}_gfp.txt", f"{lower_prefix}_lamina.txt"),
        "mcherry": first_of(f"{lower_prefix}_mcherry.txt", f"{lower_prefix}_chromatin.txt"),
        "mask": first_of(f"{lower_prefix}_mask.txt"),
        "param": first_of(f"{lower_prefix}_parameters.txt", f"{lower_prefix}_parameter.txt"),
    }
    notes = {"gfp": None, "mcherry": None, "mask": None, "param": None}

    for key in ("gfp", "mcherry", "mask", "param"):
        if result[key] is None:
            fname, note = _fuzzy_channel_match(prefix, _CHANNEL_ALTS[key], file_list)
            if fname is not None:
                result[key] = fname
                notes[key] = note

    result["notes"] = notes
    return result


def dataset_note(channel_files):
    """Combine any per-channel annotation notes (see find_channel_files) into
    one human-readable warning string, e.g. "mCherry: treated weak one".
    None if no channel needed a fuzzy match to be found."""
    notes = channel_files.get("notes") or {}
    labels = {"gfp": "GFP", "mcherry": "mCherry", "mask": "Mask", "param": "Parameters"}
    parts = [f"{labels[k]}: {v}" for k, v in notes.items() if v]
    return "; ".join(parts) if parts else None


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
    """Read a flat-text intensity file into a 1D float array.

    Real LabVIEW exports are tens of MB (400x400 pixels x dozens of frames,
    whitespace-separated per frame row). np.loadtxt parses that ~3x faster
    than a regex-findall over the raw text, with identical output, and adds
    no new dependency.
    """
    return np.loadtxt(path, dtype=np.float64).ravel()


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
