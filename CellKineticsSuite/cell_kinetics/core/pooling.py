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
    if 'Excluded' in df.columns:
        # A dataset excluded via the data-curation window -- e.g. a trial
        # with a bad segmentation or an outlier control that's inflating
        # the pooled spread. Missing/blank means "included", same as a CSV
        # written before this column existed.
        df = df[~df['Excluded'].fillna(False).astype(bool)]
        if df.empty:
            return None, ""

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


def leave_one_out_outlier_flags(k_values_by_id):
    """Given one group's replicate k-values (dict id -> k), find which
    replicate(s) would shrink the group's spread the most if excluded.

    This is a leave-one-out diagnostic, not "how far is this from the
    mean" -- a raw distance-from-mean is uninformative at N=2, since both
    points are equidistant from the mean by construction and neither one
    can be picked out as "the" outlier that way. Removing each point in
    turn and comparing the resulting std directly answers "which one is
    the best candidate to exclude."

    Only flags anything when the group's current std exceeds
    STD_K_OUTLIER_THRESHOLD -- a tight group has no outlier to find. Among
    replicates whose removal would help, only the one(s) within 90% of the
    single best improvement are returned, so a clear lone outlier is
    singled out while a genuine near-tie (e.g. N=2, where both points are
    equally "the" outlier) is reported as such rather than picking one
    arbitrarily.

    Returns {id: (current_std, std_without_this_one)} for the flagged
    replicate(s); empty if the group isn't high-spread or has fewer than
    2 valid (non-NaN) values.
    """
    ids = [i for i, v in k_values_by_id.items() if pd.notna(v)]
    if len(ids) < 2:
        return {}

    values = pd.Series({i: k_values_by_id[i] for i in ids})
    current_std = values.std()
    if pd.isna(current_std) or current_std <= STD_K_OUTLIER_THRESHOLD:
        return {}

    deltas = {}
    for i in ids:
        rest = values.drop(i)
        new_std = rest.std() if len(rest) > 1 else 0.0
        delta = current_std - new_std
        if delta > 0:
            deltas[i] = (delta, new_std)

    if not deltas:
        return {}
    best_delta = max(d for d, _ in deltas.values())
    return {i: (current_std, new_std) for i, (d, new_std) in deltas.items() if d >= 0.9 * best_delta}


def get_available_wavelengths(grouped_df):
    """Wavelengths that have at least one replicate with a known laser power."""
    if grouped_df is None or grouped_df.empty:
        return []
    with_power = grouped_df[grouped_df['Power_Num'].notna()]
    return sorted(with_power['Wavelength_Num'].dropna().unique().tolist())
