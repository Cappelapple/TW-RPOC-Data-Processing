"""OpenCV-based classic segmentation methods, adapted from
FishyIshySegmentation.py, as an alternative to Cellpose.

Same contract as segmentation.run_segmentation: takes a 2D image, returns an
integer label array (0 = background, 1..N = individual regions). That shared
convention is what lets classify_cells() work identically regardless of
which backend produced the labels -- see gui/main_window.py's
run_auto_segmentation, which picks a backend and otherwise treats the result
the same way either way.

These methods are CPU-only and fast (no GPU, no model weights), which makes
them a reasonable fallback when Cellpose isn't available or when cells are
bright/well-separated enough that a classic threshold does just as well.
"""
import cv2
import numpy as np

METHOD_THRESHOLD_COMPONENTS = "Threshold + Components"
METHOD_WATERSHED = "Watershed"
METHOD_ADAPTIVE_THRESHOLD = "Adaptive Threshold"
CLASSIC_METHODS = [METHOD_THRESHOLD_COMPONENTS, METHOD_WATERSHED, METHOD_ADAPTIVE_THRESHOLD]


def _to_u8(image):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("segment expects a 2D image")
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - mn) / (mx - mn)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _apply_morph(mask, open_ksize=3, close_ksize=9):
    result = mask.astype(np.uint8)
    if open_ksize > 1:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, k_open)
    if close_ksize > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k_close)
    return result


def _components_to_labels(mask, min_area=120):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        filtered[labels == idx] = next_id
        next_id += 1
    return filtered


def _threshold_components(image, low=0.2, high=0.8, min_area=120, open_ksize=3, close_ksize=9):
    img_u8 = _to_u8(image)
    lo = int(max(0, min(255, round(low * 255.0))))
    hi = int(max(0, min(255, round(high * 255.0))))
    if hi < lo:
        lo, hi = hi, lo
    mask = cv2.inRange(img_u8, lo, hi)
    mask = _apply_morph(mask, open_ksize=open_ksize, close_ksize=close_ksize)
    return _components_to_labels(mask, min_area=min_area)


def _watershed(image, low=0.2, high=0.8, min_area=120, min_distance=10, open_ksize=3, close_ksize=9):
    base = _threshold_components(
        image, low=low, high=high, min_area=max(1, min_area // 2),
        open_ksize=open_ksize, close_ksize=close_ksize,
    )
    sure_fg = np.uint8(base > 0) * 255
    if np.count_nonzero(sure_fg) == 0:
        return np.zeros(base.shape, dtype=np.int32)

    dist = cv2.distanceTransform(sure_fg, cv2.DIST_L2, 5)
    _, peaks = cv2.threshold(dist, float(min_distance), 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    n_markers, markers = cv2.connectedComponents(peaks)
    if n_markers <= 1:
        return _components_to_labels(sure_fg, min_area=min_area)

    markers = markers + 1
    markers[sure_fg == 0] = 0
    image_u8 = _to_u8(image)
    image_rgb = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(image_rgb, markers)
    labels = np.where(markers > 1, markers - 1, 0).astype(np.int32)

    filtered = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for label_id in np.unique(labels):
        if label_id <= 0:
            continue
        region = labels == label_id
        if int(np.count_nonzero(region)) < min_area:
            continue
        filtered[region] = next_id
        next_id += 1
    return filtered


def _adaptive_threshold(image, block_size=61, C=-5, min_area=120, open_ksize=3, close_ksize=9, low=None, high=None):
    # low/high accepted (and ignored) so callers can pass the same kwargs to
    # every classic method uniformly -- adaptive thresholding computes its
    # own local threshold per-pixel and has no fixed global intensity band.
    img_u8 = _to_u8(image)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(3, block_size)
    mask = cv2.adaptiveThreshold(
        img_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C,
    )
    mask = _apply_morph(mask, open_ksize=open_ksize, close_ksize=close_ksize)
    return _components_to_labels(mask, min_area=min_area)


def run_classic_segmentation(image, method, **kwargs):
    """Dispatch to one of the classic methods by name. Same return
    convention as segmentation.run_segmentation: an integer label array."""
    name = method.strip().lower()
    if name == METHOD_THRESHOLD_COMPONENTS.lower():
        return _threshold_components(image, **kwargs)
    if name == METHOD_WATERSHED.lower():
        return _watershed(image, **kwargs)
    if name == METHOD_ADAPTIVE_THRESHOLD.lower():
        return _adaptive_threshold(image, **kwargs)
    raise ValueError(f"unknown segmentation method '{method}'")
