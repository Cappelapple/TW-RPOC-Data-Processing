import os
import numpy as np

from Oldversionsandtesters.QuickCheck.quick_check import normalized_region_trace, load_partial_stack


def test_normalized_region_trace_basic_decay():
    stack = np.array([
        [[10, 10], [1, 1]],
        [[5, 5], [1, 1]],
    ], dtype=float)
    mask = np.array([[True, True], [False, False]])
    trace = normalized_region_trace(stack, mask)
    assert np.allclose(trace, [1.0, 0.5])


def test_normalized_region_trace_empty_mask_is_nan():
    stack = np.zeros((3, 2, 2))
    mask = np.zeros((2, 2), dtype=bool)
    trace = normalized_region_trace(stack, mask)
    assert np.all(np.isnan(trace))


def test_normalized_region_trace_zero_f0_is_nan():
    stack = np.array([[[0, 0]], [[5, 5]]], dtype=float)
    mask = np.array([[True, True]])
    trace = normalized_region_trace(stack, mask)
    assert np.all(np.isnan(trace))


def test_load_partial_stack_uses_only_complete_frames(tmp_path):
    # 2x2 = 4 values per frame; write 1.5 frames worth of numbers -- the
    # trailing partial frame must be dropped, not raise or get padded.
    path = tmp_path / "partial_GFP.txt"
    values = list(range(6))  # 1 full frame (4 values) + 2 leftover
    path.write_text(" ".join(str(v) for v in values))

    stack, num_frames = load_partial_stack(str(path), x_pixels=2, y_pixels=2)
    assert num_frames == 1
    assert stack.shape == (1, 2, 2)
    assert np.array_equal(stack[0], [[0, 1], [2, 3]])


def test_load_partial_stack_raises_with_no_complete_frame(tmp_path):
    path = tmp_path / "tiny_GFP.txt"
    path.write_text("1 2 3")
    try:
        load_partial_stack(str(path), x_pixels=10, y_pixels=10)
        assert False, "expected ValueError"
    except ValueError:
        pass
