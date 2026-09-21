"""Group replicate runs and compute pooled statistics.

Reading the summary CSV off disk is a persistence concern (see
core/persistence.py::read_summary_csv); this module only works with an
already-loaded DataFrame, so it's trivially testable with synthetic data.
"""
import numpy as np
import pandas as pd

STD_K_OUTLIER_THRESHOLD = 0.05


def pool_replicates(df):
    """Group by Wavelength+Power+Condition+Fluorophore and compute
    mean/std/count of Decay_Constant_k for each group.

    Returns (grouped_df, anomaly_text). grouped_df is None if df is None or
    empty. anomaly_text lists any group whose std exceeds
    STD_K_OUTLIER_THRESHOLD, for surfacing a "high spread" warning to the user.
    """
    if df is None or df.empty:
        return None, ""

    df = df.copy()
    df['Wavelength_Num'] = df['Wavelength'].astype(str).str.extract(r'(\d+)').astype(float)

    if 'Power_mW' in df.columns:
        df['Power_Num'] = df['Power_mW'].astype(str).str.extract(r'(\d+(?:\.\d+)?)').astype(float)
    else:
        df['Power_Num'] = np.nan

    df = df.dropna(subset=['Wavelength_Num', 'Decay_Constant_k'])
    if df.empty:
        return None, ""

    grouped = df.groupby(["Wavelength_Num", "Power_Num", "Condition", "Fluorophore"], dropna=False).agg(
        Mean_k=('Decay_Constant_k', 'mean'),
        Std_k=('Decay_Constant_k', 'std'),
        Count=('Decay_Constant_k', 'count')
    ).reset_index()
    grouped['Std_k'] = grouped['Std_k'].fillna(0.0)

    anomaly_msg = ""
    outliers = grouped[grouped['Std_k'] > STD_K_OUTLIER_THRESHOLD]
    if not outliers.empty:
        lines = []
        for _, r in outliers.iterrows():
            pwr_str = f"{r['Power_Num']:g}mW" if pd.notna(r['Power_Num']) else "power unknown"
            lines.append(
                f"• {r['Fluorophore']} ({int(r['Wavelength_Num'])}nm, {pwr_str} - {r['Condition']}): "
                f"High spread (σ={r['Std_k']:.4f}, N={r['Count']})"
            )
        anomaly_msg = "\n".join(lines)

    return grouped, anomaly_msg


def get_available_wavelengths(grouped_df):
    """Wavelengths that have at least one replicate with a known laser power."""
    if grouped_df is None or grouped_df.empty:
        return []
    with_power = grouped_df[grouped_df['Power_Num'].notna()]
    return sorted(with_power['Wavelength_Num'].dropna().unique().tolist())
