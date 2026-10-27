# ReproducibilityCheck

A bare-bones, no-GUI tool for one question: **across multiple days of
imaging, how reproducible are the results, and which trials are dragging
down the spread?**

It combines datasets from several folders (one per day, typically),
computes a decay constant per trial, pools replicates by
wavelength/power/condition/fluorophore across all folders combined, and
automatically flags + excludes whichever replicate(s) in each group would
shrink that group's spread the most -- the same leave-one-out diagnostic
CellKineticsSuite's Manage Data window uses interactively, just applied
automatically here since there's no UI to click a checkbox.

This file is completely separate from `QuickCheck/` and from
`CellKineticsSuite/` -- it doesn't import either, and only *reads*
CellKineticsSuite's core modules (file-matching, mask-threshold, fitting,
pooling, plotting) to reuse already-tested logic. Nothing in either of
those folders is modified by running this.

## A real simplification, worth knowing before you trust the numbers

Like QuickCheck, this skips per-cell segmentation entirely:

- **Treated** = the laser-target mask region.
- **Control** = literally everything else in the frame -- real control
  cells and background mixed together, not classified control cells the
  way CellKineticsSuite's real pipeline finds them.

The division step that cancels out imaging-induced bleaching
(`Treated_norm / Rest_norm`) is the same math the real pipeline uses, just
applied to this coarser region. Good for a fast reproducibility sanity
check across days; not a replacement for the real per-cell analysis when
you need the final numbers.

## Usage

```
py -3.12 reproducibility_check.py <x_pixels> <y_pixels> <day1_folder> [<day2_folder> ...]
```

Prints, per flagged outlier, which replicate got excluded and how much it
would shrink that group's spread -- then a before/after reproducibility
report (Mean k ± Std k per wavelength/power/condition/fluorophore).

Writes to the **first** folder given:

- `reproducibility_summary.csv` -- one row per dataset/channel found
  across every folder, with `Excluded`/`ExcludeReason` columns (lab-
  annotated files are excluded automatically too, same as
  CellKineticsSuite's auto-run)
- `reproducibility_report.png` -- pooled k vs. wavelength (post-exclusion),
  error bars from Std_k

## Tests

```
py -3.12 -m pytest test_reproducibility_check.py -v
```
