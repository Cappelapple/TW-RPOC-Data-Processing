import numpy as np

from cell_kinetics.core.classic_segmentation import (
    METHOD_ADAPTIVE_THRESHOLD,
    METHOD_THRESHOLD_COMPONENTS,
    METHOD_WATERSHED,
    run_classic_segmentation,
)


def make_two_blob_image(size=100):
    yy, xx = np.mgrid[:size, :size]

    def blob(cy, cx, r, amp):
        return amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))

    img = 20 + blob(25, 25, 10, 200) + blob(75, 75, 10, 200)
    return img.astype(np.float32)


def test_threshold_components_finds_two_blobs():
    img = make_two_blob_image()
    labels = run_classic_segmentation(img, METHOD_THRESHOLD_COMPONENTS, low=0.3, high=1.0, min_area=20)
    assert labels.shape == img.shape
    assert labels.max() >= 2


def test_watershed_finds_two_blobs():
    img = make_two_blob_image()
    labels = run_classic_segmentation(img, METHOD_WATERSHED, low=0.3, high=1.0, min_area=20, min_distance=3)
    assert labels.max() >= 2


def test_adaptive_threshold_runs_without_error():
    img = make_two_blob_image()
    labels = run_classic_segmentation(img, METHOD_ADAPTIVE_THRESHOLD, min_area=10)
    assert labels.shape == img.shape


def test_blank_image_yields_no_labels():
    img = np.full((50, 50), 20.0, dtype=np.float32)
    labels = run_classic_segmentation(img, METHOD_THRESHOLD_COMPONENTS, low=0.3, high=1.0, min_area=20)
    assert labels.max() == 0


def test_unknown_method_raises():
    img = make_two_blob_image()
    try:
        run_classic_segmentation(img, "not a real method")
        assert False, "expected ValueError"
    except ValueError:
        pass
