import pandas as pd

from cell_kinetics.core.pooling import get_available_wavelengths, pool_replicates


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


def test_pool_replicates_handles_empty_and_none():
    grouped, anomaly = pool_replicates(None)
    assert grouped is None
    assert anomaly == ""

    grouped2, anomaly2 = pool_replicates(pd.DataFrame())
    assert grouped2 is None
    assert anomaly2 == ""


def test_get_available_wavelengths_only_returns_power_tagged():
    df = make_summary_df([
        {"Dataset": "A", "Wavelength": "1000nm", "Condition": "Normoxia", "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.10},
        {"Dataset": "B", "Wavelength": "900nm", "Condition": "Normoxia", "Power_mW": "Unknown", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.12},
    ])
    grouped, _ = pool_replicates(df)

    wavelengths = get_available_wavelengths(grouped)
    assert wavelengths == [1000.0]
