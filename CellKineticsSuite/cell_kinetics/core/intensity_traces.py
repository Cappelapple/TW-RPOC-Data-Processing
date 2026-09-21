"""Background-subtract -> normalize -> control-correct pipeline.

This is a standard FRAP/photobleaching kinetics pipeline: for each frame,
average intensity within a region, subtract the background region's average,
normalize to the first frame (F/F0), and optionally divide by a control
region's F/F0 to cancel out generic imaging-induced bleaching that isn't
specific to the treatment.

Whether to use a locally-drawn control, a "borrowed" one from a shared
control library, or none at all is a policy decision that depends on
persisted, stateful data -- that decision is made by the caller (see
gui/main_window.py), which passes the resolved control trace in here via
fallback_gfp_control_norm/fallback_mch_control_norm. This module only knows
about arrays, not about the shared-control-library dict.
"""
import numpy as np


def mask_average_trace(stack, masks_list, num_frames):
    """Per-frame mean intensity across every mask in masks_list, then
    averaged across masks (unweighted -- a small cell counts as much as a
    large one). Frames with no valid mask default to 1.0."""
    intensities = []
    for frame in stack[:num_frames]:
        frame_vals = [np.mean(frame[m]) for m in masks_list if np.any(m)]
        intensities.append(np.mean(frame_vals) if frame_vals else 1.0)
    return np.array(intensities)


def net_normalized_trace(stack, target_masks, background_masks, num_frames):
    """Background-subtracted, F/F0-normalized trace for one region."""
    raw = mask_average_trace(stack, target_masks, num_frames)
    bg = mask_average_trace(stack, background_masks, num_frames)
    net = np.maximum(raw - bg, 1e-4)
    return net / net[0]


def correct_with_control(target_norm, control_norm):
    """Divide a normalized target trace by a normalized control trace."""
    return target_norm / np.clip(control_norm, 1e-4, None)


def total_mask_pixels(masks_list):
    """Total pixel count across every mask in the list (used for the
    auto-run pixel-count validity gate)."""
    return sum(int(np.sum(m)) for m in masks_list)


def compute_corrected_traces(
    gfp_stack, mcherry_stack, treated_masks, control_masks, background_masks, num_frames,
    fallback_gfp_control_norm=None, fallback_mch_control_norm=None,
):
    """Full pipeline for one dataset's treated region, for both channels.

    fallback_*_control_norm: pass a pre-computed normalized control trace
    (e.g. borrowed from a shared-control library) to use when control_masks
    is empty. Leave None to fit without photobleach correction in that case.

    Returns a dict with:
      gfp_t_corr, mch_t_corr: corrected treated traces (what gets fit)
      gfp_control_norm, mch_control_norm: the control trace actually used
        (for plotting the dashed reference line), or None
      has_gfp_control, has_mch_control: bool
      borrowed_control: True if either channel used the fallback rather
        than a locally-drawn control
    """
    gfp_t_norm = net_normalized_trace(gfp_stack, treated_masks, background_masks, num_frames)
    mch_t_norm = net_normalized_trace(mcherry_stack, treated_masks, background_masks, num_frames)

    borrowed_control = False

    if control_masks:
        gfp_control_norm = net_normalized_trace(gfp_stack, control_masks, background_masks, num_frames)
        has_gfp_control = True
    elif fallback_gfp_control_norm is not None:
        gfp_control_norm = fallback_gfp_control_norm
        has_gfp_control = True
        borrowed_control = True
    else:
        gfp_control_norm = None
        has_gfp_control = False

    if control_masks:
        mch_control_norm = net_normalized_trace(mcherry_stack, control_masks, background_masks, num_frames)
        has_mch_control = True
    elif fallback_mch_control_norm is not None:
        mch_control_norm = fallback_mch_control_norm
        has_mch_control = True
        borrowed_control = True
    else:
        mch_control_norm = None
        has_mch_control = False

    gfp_t_corr = correct_with_control(gfp_t_norm, gfp_control_norm) if has_gfp_control else gfp_t_norm
    mch_t_corr = correct_with_control(mch_t_norm, mch_control_norm) if has_mch_control else mch_t_norm

    return {
        "gfp_t_norm": gfp_t_norm,
        "mch_t_norm": mch_t_norm,
        "gfp_t_corr": gfp_t_corr,
        "mch_t_corr": mch_t_corr,
        "gfp_control_norm": gfp_control_norm,
        "mch_control_norm": mch_control_norm,
        "has_gfp_control": has_gfp_control,
        "has_mch_control": has_mch_control,
        "borrowed_control": borrowed_control,
    }
