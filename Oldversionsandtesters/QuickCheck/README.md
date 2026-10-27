# QuickCheck

A bare-bones, no-GUI validity check for a photobleaching run: no cell
segmentation, no control-division, no curve fitting. It just plots two
normalized curves per channel -- the average intensity inside the
laser-target mask ("Treated") and the average intensity of everything else
in the frame ("Rest") -- so you can glance at a trial while you're still at
the microscope and tell whether it actually bleached where it was supposed
to.

Treated should visibly drop; Rest should stay roughly flat. If they don't
look like that, something's wrong with the trial (mask misaligned, laser
power off, wrong region selected) and it's worth re-running before you move
on, instead of finding out during full analysis later.

This is **not** the real analysis -- for per-cell segmentation, treated
vs. control classification, photobleach correction, and curve fitting, use
`CellKineticsSuite` (a sibling folder this tool reuses two small pieces
from: file-matching and mask-threshold logic, both plain numpy).

## Usage

```
py -3.12 quick_check.py "<folder>" <x_pixels> <y_pixels>
```

Points at a working folder and reports on every dataset (GFP/mCherry/Mask
file triplet) it finds there. `x_pixels`/`y_pixels` are the two values
needed to reshape a flat-text channel file correctly -- they're asked for
explicitly instead of being read from a Parameters.txt because this is
meant to work on a **still-growing file**: it uses however many complete
frames exist right now, so you can check a long acquisition partway through
instead of waiting for it to finish. Pixel dimensions are fixed from the
moment a scan starts; frame count isn't.

For each dataset, it prints a percent-drop summary per channel and a
go/no-go verdict, and saves `<prefix>_quick_check.png` (Treated vs. Rest,
both channels) into the same folder.

## Tests

```
py -3.12 -m pytest test_quick_check.py -v
```
