"""Disk persistence: session cache, shared-control library, the pooled
summary CSV, and session archiving.

All functions take an explicit folder path rather than reading app state,
so they're usable outside the GUI (e.g. a standalone script that just wants
to read `master_kinetics_summary.csv` from a given results folder).
"""
import os
import pickle
import shutil
import time

import pandas as pd

CACHE_FILENAME = "session_traces_cache.pkl"
LIBRARY_FILENAME = "shared_control_library.pkl"
SUMMARY_FILENAME = "master_kinetics_summary.csv"
SESSION_FILES = [SUMMARY_FILENAME, CACHE_FILENAME, LIBRARY_FILENAME]


def save_session_cache(folder, traces_cache, control_library):
    """Write both the per-run traces cache and the shared-control library.
    Raises on failure -- callers decide how to surface that."""
    with open(os.path.join(folder, CACHE_FILENAME), "wb") as f:
        pickle.dump(traces_cache, f)
    with open(os.path.join(folder, LIBRARY_FILENAME), "wb") as f:
        pickle.dump(control_library, f)


def load_session_cache(folder):
    """Returns (traces_cache, control_library), defaulting to {} for any
    file that's missing or fails to unpickle."""
    traces_cache, control_library = {}, {}

    cache_path = os.path.join(folder, CACHE_FILENAME)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                traces_cache = pickle.load(f)
        except Exception as e:
            print(f"Error loading session cache: {e}")

    lib_path = os.path.join(folder, LIBRARY_FILENAME)
    if os.path.exists(lib_path):
        try:
            with open(lib_path, "rb") as f:
                control_library = pickle.load(f)
        except Exception as e:
            print(f"Error loading control library: {e}")

    return traces_cache, control_library


def read_summary_csv(folder):
    """The pooled results CSV as a DataFrame, or None if it doesn't exist
    yet / is too small to be meaningful."""
    summary_path = os.path.join(folder, SUMMARY_FILENAME)
    if not os.path.exists(summary_path) or os.path.getsize(summary_path) < 10:
        return None
    return pd.read_csv(summary_path)


def append_summary_rows(folder, dataset_name, new_rows):
    """Replace any existing rows for dataset_name with new_rows and write
    the summary CSV back out. Raises on failure."""
    summary_path = os.path.join(folder, SUMMARY_FILENAME)
    new_df = pd.DataFrame(new_rows)

    if os.path.exists(summary_path):
        master_df = pd.read_csv(summary_path)
        if "Power_mW" not in master_df.columns:
            master_df["Power_mW"] = "Unknown"
        master_df = master_df[master_df["Dataset"] != dataset_name]
        master_df = pd.concat([master_df, new_df], ignore_index=True)
    else:
        master_df = new_df

    master_df.to_csv(summary_path, index=False)


def write_summary_csv(folder, df):
    """Overwrite the summary CSV with df exactly as given. For editing
    existing rows in place -- e.g. toggling the data-curation window's
    Excluded flag -- as opposed to append_summary_rows, which only knows
    how to replace-by-dataset-name when adding newly processed rows."""
    summary_path = os.path.join(folder, SUMMARY_FILENAME)
    df.to_csv(summary_path, index=False)


def archive_session_files(folder):
    """Move any existing session-cache/summary files into a timestamped
    Archived_Sessions subfolder (never deletes anything).

    Returns (archived_filenames, archive_dir). archive_dir is None if there
    was nothing to archive.
    """
    existing = [f for f in SESSION_FILES if os.path.exists(os.path.join(folder, f))]
    if not existing:
        return [], None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(folder, "Archived_Sessions", timestamp)
    os.makedirs(archive_dir, exist_ok=True)
    for f in existing:
        shutil.move(os.path.join(folder, f), os.path.join(archive_dir, f))

    return existing, archive_dir
