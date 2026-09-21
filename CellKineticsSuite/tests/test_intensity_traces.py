import numpy as np

from cell_kinetics.core.intensity_traces import (
    compute_corrected_traces,
    correct_with_control,
    mask_average_trace,
    net_normalized_trace,
    total_mask_pixels,
)


def make_synthetic_stack(num_frames, size, region_mask, start_value, decay_rate, floor=10.0):
    """A stack where region_mask decays exponentially from start_value down to
    floor, and everywhere else stays at floor."""
    stack = np.full((num_frames, size, size), floor, dtype=float)
    for t in range(num_frames):
        stack[t][region_mask] = floor + (start_value - floor) * np.exp(-decay_rate * t)
    return stack


def test_mask_average_trace_basic():
    size, frames = 10, 5
    mask = np.zeros((size, size), dtype=bool)
    mask[2:4, 2:4] = True
    stack = make_synthetic_stack(frames, size, mask, start_value=100, decay_rate=0.5)

    trace = mask_average_trace(stack, [mask], num_frames=frames)
    assert len(trace) == frames
    assert trace[0] > trace[-1]  # decaying


def test_mask_average_trace_defaults_when_no_valid_mask():
    stack = np.zeros((3, 5, 5))
    empty_mask = np.zeros((5, 5), dtype=bool)
    trace = mask_average_trace(stack, [empty_mask], num_frames=3)
    assert np.all(trace == 1.0)


def test_net_normalized_trace_starts_at_one():
    size, frames = 10, 6
    target = np.zeros((size, size), dtype=bool)
    target[1:3, 1:3] = True
    bg = np.zeros((size, size), dtype=bool)
    bg[8:10, 8:10] = True
    stack = make_synthetic_stack(frames, size, target, start_value=150, decay_rate=0.4, floor=20.0)

    norm = net_normalized_trace(stack, [target], [bg], frames)
    assert np.isclose(norm[0], 1.0)
    assert norm[-1] < norm[0]  # still decaying after background subtraction


def test_correct_with_control_cancels_identical_bleaching():
    target_norm = np.array([1.0, 0.8, 0.6, 0.5])
    # If the control bleaches identically to the target, the corrected trace
    # should be flat at 1.0 -- there's no treatment-specific decay left.
    control_norm = target_norm.copy()
    corrected = correct_with_control(target_norm, control_norm)
    assert np.allclose(corrected, 1.0)


def test_total_mask_pixels_sums_across_masks():
    m1 = np.zeros((5, 5), dtype=bool)
    m1[0, 0] = True
    m2 = np.zeros((5, 5), dtype=bool)
    m2[1, 1] = m2[1, 2] = True
    assert total_mask_pixels([m1, m2]) == 3


def test_compute_corrected_traces_with_local_control():
    size, frames = 20, 6
    treated = np.zeros((size, size), dtype=bool)
    treated[1:4, 1:4] = True
    control = np.zeros((size, size), dtype=bool)
    control[10:13, 10:13] = True
    bg = np.zeros((size, size), dtype=bool)
    bg[15:18, 15:18] = True

    gfp_stack = make_synthetic_stack(frames, size, treated, 200, 0.5, floor=10)
    for t in range(frames):
        gfp_stack[t][control] = 10 + (150 - 10) * np.exp(-0.1 * t)  # slower bleach = "real" control
    mch_stack = gfp_stack.copy()

    result = compute_corrected_traces(gfp_stack, mch_stack, [treated], [control], [bg], frames)

    assert result["has_gfp_control"]
    assert result["has_mch_control"]
    assert not result["borrowed_control"]
    assert np.isclose(result["gfp_t_corr"][0], 1.0)
    # treated bleaches faster than control, so the corrected trace should still decay
    assert result["gfp_t_corr"][-1] < result["gfp_t_corr"][0]


def test_compute_corrected_traces_without_control_or_fallback():
    size, frames = 10, 5
    treated = np.zeros((size, size), dtype=bool)
    treated[1:3, 1:3] = True
    bg = np.zeros((size, size), dtype=bool)
    bg[7:9, 7:9] = True
    stack = make_synthetic_stack(frames, size, treated, 100, 0.3)

    result = compute_corrected_traces(stack, stack, [treated], [], [bg], frames)

    assert not result["has_gfp_control"]
    assert not result["has_mch_control"]
    assert not result["borrowed_control"]
    # uncorrected trace equals the plain normalized trace
    assert np.allclose(result["gfp_t_corr"], result["gfp_t_norm"])


def test_compute_corrected_traces_uses_fallback_when_no_local_control():
    size, frames = 10, 5
    treated = np.zeros((size, size), dtype=bool)
    treated[1:3, 1:3] = True
    bg = np.zeros((size, size), dtype=bool)
    bg[7:9, 7:9] = True
    stack = make_synthetic_stack(frames, size, treated, 100, 0.3)

    fallback = np.linspace(1.0, 0.9, frames)  # a "borrowed" control trace
    result = compute_corrected_traces(
        stack, stack, [treated], [], [bg], frames,
        fallback_gfp_control_norm=fallback, fallback_mch_control_norm=fallback,
    )

    assert result["has_gfp_control"]
    assert result["borrowed_control"]
    assert not np.allclose(result["gfp_t_corr"], result["gfp_t_norm"])
