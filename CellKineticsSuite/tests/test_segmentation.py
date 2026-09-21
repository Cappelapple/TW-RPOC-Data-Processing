import numpy as np

from cell_kinetics.core.segmentation import (
    build_laser_mask,
    classify_cells,
    polygon_to_mask,
    refine_mask_by_local_threshold,
)


def make_two_cell_labels(size=60):
    """label 1 in the top-left quadrant, label 2 in the bottom-right quadrant."""
    labels = np.zeros((size, size), dtype=int)
    labels[5:25, 5:25] = 1
    labels[35:55, 35:55] = 2
    return labels


def test_build_laser_mask_thresholds_relative_to_peak():
    frame = np.zeros((10, 10))
    frame[2:4, 2:4] = 200  # bright laser spot
    mask = build_laser_mask(frame, frame.shape)
    assert mask[3, 3]
    assert not mask[0, 0]


def test_build_laser_mask_all_false_when_no_reference():
    mask = build_laser_mask(None, (5, 5))
    assert not mask.any()


def test_classify_cells_overlapping_becomes_treated_nonoverlapping_becomes_control():
    labels = make_two_cell_labels()
    laser_mask = np.zeros_like(labels, dtype=bool)
    laser_mask[0:30, 0:30] = True  # overlaps cell 1 only

    result = classify_cells(labels, laser_mask, min_area=1, overlap_frac_thresh=0.3, auto_control_enabled=True)

    assert result["n_treated"] == 1
    assert result["n_control"] == 1
    assert result["n_excluded"] == 0
    # background excludes both cells
    assert not result["background_mask"][10, 10]  # inside cell 1
    assert not result["background_mask"][45, 45]  # inside cell 2
    assert result["background_mask"][0, 59]  # untouched corner


def test_classify_cells_auto_control_disabled_excludes_but_still_removes_from_background():
    labels = make_two_cell_labels()
    laser_mask = np.zeros_like(labels, dtype=bool)
    laser_mask[0:30, 0:30] = True

    result = classify_cells(labels, laser_mask, min_area=1, overlap_frac_thresh=0.3, auto_control_enabled=False)

    assert result["n_treated"] == 1
    assert result["n_control"] == 0
    assert result["n_excluded"] == 1
    # the excluded (non-treated) cell is still cut out of the background estimate
    assert not result["background_mask"][45, 45]


def test_classify_cells_drops_small_cells():
    labels = make_two_cell_labels()  # each cell is 20x20 = 400px
    laser_mask = np.zeros_like(labels, dtype=bool)

    result = classify_cells(labels, laser_mask, min_area=1000, overlap_frac_thresh=0.3, auto_control_enabled=True)

    assert result["n_dropped"] == 2
    assert result["n_treated"] == 0
    assert result["n_control"] == 0


def test_polygon_to_mask_rasterizes_a_square():
    verts = [(2, 2), (2, 8), (8, 8), (8, 2)]
    mask = polygon_to_mask(verts, y_pixels=10, x_pixels=10)
    assert mask[5, 5]  # center of the square
    assert not mask[0, 0]  # outside


def test_refine_mask_by_local_threshold_keeps_only_brightest_pixels():
    frame = np.zeros((10, 10))
    frame[3, 3] = 10.0
    frame[3, 4] = 100.0
    raw_mask = np.zeros((10, 10), dtype=bool)
    raw_mask[3, 3] = True
    raw_mask[3, 4] = True

    refined = refine_mask_by_local_threshold(frame, raw_mask, thresh_pct=50)
    assert refined[3, 4]
    assert not refined[3, 3]


def test_refine_mask_by_local_threshold_falls_back_when_candidate_empty():
    # A threshold so strict nothing clears it should just return raw_mask unchanged.
    frame = np.ones((5, 5))
    raw_mask = np.zeros((5, 5), dtype=bool)
    raw_mask[2, 2] = True
    refined = refine_mask_by_local_threshold(frame, raw_mask, thresh_pct=100)
    assert np.array_equal(refined, raw_mask)
