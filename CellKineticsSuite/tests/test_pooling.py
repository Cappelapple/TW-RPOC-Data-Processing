import numpy as np
import pandas as pd

from cell_kinetics.core.pooling import get_available_wavelengths, leave_one_out_outlier_flags, pool_replicates


def make_summary_df(rows):
    return pd.DataFrame(rows)


def test_pool_replicates_groups_and_averages():
    df = make_summary_df([
        {"Dataset": "A", "Wavelength": "1000nm", "Condition": "Normoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.10},
        {"Dataset": "B", "Wavelength": "1000nm", "Condition": "Normoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.12},
        {"Dataset": "C", "Wavelength": "1000nm", "Condition": "Hypoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.20},
    ])

    grouped, anomaly = pool_replicates(df)

    assert grouped is not None
    normoxia_row = grouped[(grouped["Condition"] == "Normoxia") & (grouped["Wavelength_Num"] == 1000)]
    assert len(normoxia_row) == 1
    assert normoxia_row.iloc[0]["Count"] == 2
    assert abs(normoxia_row.iloc[0]["Mean_k"] - 0.11) < 1e-9


def test_pool_replicates_flags_high_spread():
    df = make_summary_df([
        {"Dataset": "A", "Wavelength": "1000nm", "Condition": "Normoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.10},
        {"Dataset": "B", "Wavelength": "1000nm", "Condition": "Normoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.50},
    ])

    grouped, anomaly = pool_replicates(df)

    assert anomaly != ""
    assert "High spread" in anomaly


def test_pool_replicates_skips_excluded_rows():
    df = make_summary_df([
        {"Dataset": "A", "Wavelength": "1000nm", "Condition": "Normoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.10, "Excluded": False},
        {"Dataset": "B", "Wavelength": "1000nm", "Condition": "Normoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.50, "Excluded": True},
    ])

    grouped, anomaly = pool_replicates(df)

    assert grouped is not None
    row = grouped[(grouped["Condition"] == "Normoxia") & (grouped["Wavelength_Num"] == 1000)]
    assert row.iloc[0]["Count"] == 1
    assert abs(row.iloc[0]["Mean_k"] - 0.10) < 1e-9
    assert anomaly == ""  # the outlier that would've flagged high spread is excluded


def test_pool_replicates_treats_missing_excluded_as_included():
    df = make_summary_df([
        {"Dataset": "A", "Wavelength": "1000nm", "Condition": "Normoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.10, "Excluded": False},
        {"Dataset": "B", "Wavelength": "1000nm", "Condition": "Normoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.12, "Excluded": None},
    ])

    grouped, _ = pool_replicates(df)
    row = grouped[(grouped["Condition"] == "Normoxia") & (grouped["Wavelength_Num"] == 1000)]
    assert row.iloc[0]["Count"] == 2


def test_pool_replicates_all_excluded_returns_none():
    df = make_summary_df([
        {"Dataset": "A", "Wavelength": "1000nm", "Condition": "Normoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.10, "Excluded": True},
    ])
    grouped, anomaly = pool_replicates(df)
    assert grouped is None
    assert anomaly == ""


def test_pool_replicates_handles_empty_and_none():
    grouped, anomaly = pool_replicates(None)
    assert grouped is None
    assert anomaly == ""

    grouped2, anomaly2 = pool_replicates(pd.DataFrame())
    assert grouped2 is None
    assert anomaly2 == ""


def test_outlier_flags_singles_out_clear_lone_outlier():
    flags = leave_one_out_outlier_flags({"A": 0.10, "B": 0.11, "C": 0.50})
    assert set(flags.keys()) == {"C"}
    current_std, new_std = flags["C"]
    assert current_std > new_std


def test_outlier_flags_empty_for_tight_group():
    flags = leave_one_out_outlier_flags({"A": 0.10, "B": 0.11, "C": 0.115})
    assert flags == {}


def test_outlier_flags_both_flagged_for_ambiguous_pair():
    # N=2: both points are equidistant from the mean, so neither can be
    # singled out as "the" outlier by removal-improvement either -- both
    # should come back flagged rather than one picked arbitrarily.
    flags = leave_one_out_outlier_flags({"A": 0.10, "B": 0.50})
    assert set(flags.keys()) == {"A", "B"}


def test_outlier_flags_ignores_nan_values():
    flags = leave_one_out_outlier_flags({"A": 0.10, "B": 0.11, "C": 0.50, "D": np.nan})
    assert "D" not in flags
    assert set(flags.keys()) == {"C"}


def test_outlier_flags_empty_with_insufficient_data():
    assert leave_one_out_outlier_flags({"A": 0.10}) == {}
    assert leave_one_out_outlier_flags({}) == {}
    assert leave_one_out_outlier_flags({"A": 0.10, "B": np.nan}) == {}


def test_get_available_wavelengths_only_returns_power_tagged():
    df = make_summary_df([
        {"Dataset": "A", "Wavelength": "1000nm", "Condition": "Normoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.10},
        {"Dataset": "B", "Wavelength": "900nm", "Condition": "Normoxia", "Power_mW": "Unknown", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.12},
    ])
    grouped, _ = pool_replicates(df)

    wavelengths = get_available_wavelengths(grouped)
    assert wavelengths == [1000.0]
