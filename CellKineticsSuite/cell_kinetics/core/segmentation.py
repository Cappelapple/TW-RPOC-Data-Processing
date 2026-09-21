"""Cellpose-based cell segmentation and treated/control/background classification.

`cellpose`/`torch` are imported lazily inside load_cellpose_model, not at
module load time -- this lets the rest of the package (and its tests) import
cleanly in an environment that doesn't have GPU torch installed.
"""
import numpy as np
from matplotlib.path import Path
from skimage.measure import find_contours


def load_cellpose_model(gpu=None, pretrained_model='cyto3'):
    """Load (and by caller convention, cache) a Cellpose model.

    gpu=None auto-detects via torch.cuda.is_available(); pass True/False to
    force one or the other. Auto-detection matters a lot in practice: on a
    machine with multiple Python installs, whichever one is missing the CUDA
    build of torch will silently fall back to CPU and take minutes per frame
    instead of seconds.
    """
    from cellpose import models
    if gpu is None:
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
    return models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)


def run_segmentation(model, image, diameter=None):
    """Run the model on a single 2D image. Returns an integer label array
    (0 = background, 1..N = individual cells)."""
    img = np.asarray(image, dtype=np.float32)
    labels, _, _ = model.eval(img, channels=[0, 0], diameter=diameter)
    return labels


def build_laser_mask(laser_ref_frame, shape, threshold_frac=0.1):
    """Boolean mask of the laser-target region: pixels above threshold_frac
    of the reference frame's peak value. Returns an all-False mask of the
    given shape if there's no usable reference frame."""
    if laser_ref_frame is not None and np.max(laser_ref_frame) > 0:
        return laser_ref_frame > (np.max(laser_ref_frame) * threshold_frac)
    return np.zeros(shape, dtype=bool)


def classify_cells(labels, laser_mask, min_area, overlap_frac_thresh, auto_control_enabled):
    """Classify each detected cell as TREATED (overlaps laser_mask above
    overlap_frac_thresh) or CONTROL (doesn't, and auto_control_enabled is
    True), dropping anything under min_area. Every kept cell -- treated,
    control, or excluded -- is removed from the background estimate.

    NOTE: the caller is expected to only call this when labels.max() > 0;
    with zero detected cells there's nothing meaningful to classify (see
    gui/main_window.py's "no cells detected" early-out, which intentionally
    leaves TREATED/CONTROL/BACKGROUND all empty rather than treating the
    whole frame as background).

    Returns a dict:
      treated_masks, control_masks: list[np.ndarray[bool]]
      background_mask: np.ndarray[bool] (True = not covered by any kept cell)
      n_treated, n_control, n_excluded, n_dropped: int
    """
    n_labels = int(labels.max())
    treated_masks, control_masks = [], []
    n_treated = n_control = n_excluded = n_dropped = 0
    all_cells_mask = np.zeros(labels.shape, dtype=bool)

    for label_id in range(1, n_labels + 1):
        cell_mask = labels == label_id
        area = int(np.sum(cell_mask))
        if area < min_area:
            n_dropped += 1
            continue

        all_cells_mask |= cell_mask
        overlap_frac = np.sum(cell_mask & laser_mask) / area

        if overlap_frac >= overlap_frac_thresh:
            treated_masks.append(cell_mask)
            n_treated += 1
        elif auto_control_enabled:
            control_masks.append(cell_mask)
            n_control += 1
        else:
            n_excluded += 1

    return {
        "treated_masks": treated_masks,
        "control_masks": control_masks,
        "background_mask": ~all_cells_mask,
        "n_treated": n_treated,
        "n_control": n_control,
        "n_excluded": n_excluded,
        "n_dropped": n_dropped,
    }


def polygon_to_mask(verts, y_pixels, x_pixels):
    """Rasterize a lasso polygon's (x, y) vertices into a boolean mask --
    the manual-selection counterpart to Cellpose's per-cell label masks."""
    y_grid, x_grid = np.mgrid[:y_pixels, :x_pixels]
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
    return Path(verts).contains_points(points).reshape((y_pixels, x_pixels))


def refine_mask_by_local_threshold(frame, raw_mask, thresh_pct):
    """Restrict raw_mask to its own brightest thresh_pct..100% of pixel
    values (a local, per-selection intensity threshold -- not a global one).
    Falls back to raw_mask unchanged if that would leave nothing selected."""
    local_pixels = frame[raw_mask]
    if local_pixels.size == 0:
        return raw_mask
    mn, mx = np.min(local_pixels), np.max(local_pixels)
    thresh_line = mn + (thresh_pct / 100.0) * (mx - mn)
    candidate_mask = np.logical_and(raw_mask, frame >= thresh_line)
    if np.sum(candidate_mask) > 0:
        return candidate_mask
    return raw_mask


def mask_to_contour_vertices(mask):
    """(x, y) vertices of the longest contour of a boolean mask, for drawing
    an outline patch. Returns None if no contour could be found."""
    try:
        contours = find_contours(mask.astype(float), 0.5)
        if contours:
            longest = max(contours, key=len)
            return np.column_stack((longest[:, 1], longest[:, 0]))
    except Exception:
        pass
    return None
