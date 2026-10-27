import numpy as np
import pandas as pd

from reproducibility_check import corrected_trace, flag_and_exclude_outliers, load_full_stack


def test_corrected_trace_cancels_shared_imaging_decay():
    # Treated decays at (imaging + action); Rest decays at imaging alone --
    # dividing them out should leave something close to the action-only
    # exponential, same algebra validated earlier for the real pipeline.
    t = np.arange(20, dtype=float)
    imaging_k, action_k = 0.05, 0.2
    treated_signal = np.exp(-(imaging_k + action_k) * t)
    rest_signal = np.exp(-imaging_k * t)

    stack = np.zeros((20, 2, 2))
    stack[:, 0, 0] = treated_signal
    stack[:, 0, 1] = treated_signal
    stack[:, 1, 0] = rest_signal
    stack[:, 1, 1] = rest_signal
    treated_mask = np.array([[True, True], [False, False]])
    rest_mask = ~treated_mask

    corrected = corrected_trace(stack, treated_mask, rest_mask)
    assert np.allclose(corrected, np.exp(-action_k * t), atol=1e-6)


def test_corrected_trace_nan_when_region_starts_at_zero():
    stack = np.zeros((5, 2, 2))
    stack[:, 0, :] = 0.0  # treated region (row 0) is zero at every frame, including frame 0
    stack[:, 1, :] = 1.0  # rest region (row 1) is fine
    treated_mask = np.array([[True, True], [False, False]])
    rest_mask = ~treated_mask
    result = corrected_trace(stack, treated_mask, rest_mask)
    assert np.all(np.isnan(result))


def test_load_full_stack_uses_only_complete_frames(tmp_path):
    path = tmp_path / "partial.txt"
    values = list(range(6))  # 2x2=4 per frame: 1 full frame + 2 leftover
    path.write_text(" ".join(str(v) for v in values))
    stack, num_frames = load_full_stack(str(path), x_pixels=2, y_pixels=2)
    assert num_frames == 1
    assert np.array_equal(stack[0], [[0, 1], [2, 3]])


def test_load_full_stack_raises_with_no_complete_frame(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text("1 2 3")
    try:
        load_full_stack(str(path), x_pixels=10, y_pixels=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def make_df(rows):
    df = pd.DataFrame(rows)
    if "Excluded" not in df.columns:
        df["Excluded"] = False
    df["Excluded"] = df["Excluded"].fillna(False).astype(bool)
    if "ExcludeReason" not in df.columns:
        df["ExcludeReason"] = ""
    df["ExcludeReason"] = df["ExcludeReason"].fillna("")
    return df


def test_flag_and_exclude_outliers_flags_clear_outlier():
    df = make_df([
        {"Dataset": "A", "SourceFolder": "day1", "Wavelength": "1000nm", "Condition": "Normoxia",
         "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.10},
        {"Dataset": "B", "SourceFolder": "day2", "Wavelength": "1000nm", "Condition": "Normoxia",
         "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.11},
        {"Dataset": "C", "SourceFolder": "day2", "Wavelength": "1000nm", "Condition": "Normoxia",
         "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.50},
    ])
    result = flag_and_exclude_outliers(df)
    assert result.loc[result["Dataset"] == "C", "Excluded"].iloc[0] == True  # noqa: E712
    assert result.loc[result["Dataset"] == "A", "Excluded"].iloc[0] == False  # noqa: E712
    assert "outlier" in result.loc[result["Dataset"] == "C", "ExcludeReason"].iloc[0]


def test_flag_and_exclude_outliers_respects_preexisting_exclusions():
    # "C" is already excluded (e.g. lab-annotated) before statistical
    # flagging runs -- it must not be un-excluded, and the leave-one-out
    # pass must only consider the still-included replicates.
    df = make_df([
        {"Dataset": "A", "SourceFolder": "day1", "Wavelength": "1000nm", "Condition": "Normoxia",
         "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.10},
        {"Dataset": "B", "SourceFolder": "day2", "Wavelength": "1000nm", "Condition": "Normoxia",
         "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.11},
        {"Dataset": "C", "SourceFolder": "day2", "Wavelength": "1000nm", "Condition": "Normoxia",
         "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.50,
         "Excluded": True, "ExcludeReason": "annotated: bad trial"},
    ])
    result = flag_and_exclude_outliers(df)
    c_row = result.loc[result["Dataset"] == "C"].iloc[0]
    assert c_row["Excluded"] == True  # noqa: E712
    assert c_row["ExcludeReason"] == "annotated: bad trial"  # unchanged, not overwritten
    assert result.loc[result["Dataset"] == "A", "Excluded"].iloc[0] == False  # noqa: E712


def test_flag_and_exclude_outliers_leaves_tight_group_alone():
    df = make_df([
        {"Dataset": "A", "SourceFolder": "day1", "Wavelength": "1000nm", "Condition": "Normoxia",
         "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.10},
        {"Dataset": "B", "SourceFolder": "day2", "Wavelength": "1000nm", "Condition": "Normoxia",
         "Power_mW": "40", "Fluorophore": "GFP-LaminA", "Decay_Constant_k": 0.11},
    ])
    result = flag_and_exclude_outliers(df)
    assert not result["Excluded"].any()
