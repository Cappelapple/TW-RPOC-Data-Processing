# CellKineticsSuite

Modular rewrite of `Cell_Analyzerv16.py`: measures GFP-Lamin A and mCherry-
chromatin photobleaching decay kinetics in HeLa cells across wavelength,
laser power, and Normoxia/Hypoxia condition, then pools replicates into
comparative plots.

Same features as the original single-file app (manual lasso selection,
Cellpose-based auto-segmentation, fully-automatic batch mode, Box background
sync, the analytics dashboard, publication export, session archiving) --
this is a structural refactor, not a rewrite. It's organized so a future
change to one piece (a new segmentation backend, a different fit model, a
new export format) means editing one file with a test to prove nothing else
silently broke, instead of tracing through a 2000-line GUI event handler to
find the two lines of actual math.

## Running it

```
py -3.12 main.py
```

Use Python 3.12, not your system default `python`, if you have multiple
Python installs. Cellpose needs GPU-enabled PyTorch to run at interactive
speed (seconds per frame instead of minutes); it's easy to end up on an
install that only has the CPU build. Check with:

```
py -3.12 -c "import torch; print(torch.cuda.is_available())"
```

## Layout

```
main.py                    Entry point (gc.disable(), theme setup, launches the app)
cell_kinetics/
  core/                     Pure Python -- no Tkinter imports anywhere here.
                            Callable from a plain script, a notebook, or a test
                            with nothing but numpy/scipy/pandas/cellpose installed.
    fitting.py              Exponential decay model + curve fitting
    metadata_parsing.py     Wavelength/power/condition from filenames & parameter files
    dataset_io.py           Find/load dataset file triplets (GFP/mCherry/Mask/Parameters)
    segmentation.py         Cellpose wrapper + treated/control/background classification
    intensity_traces.py     Background-subtract -> normalize -> control-correct pipeline
    pooling.py               Group replicates, compute pooled stats
    persistence.py           Session cache, summary CSV, session archiving
    plotting.py               Build matplotlib Figure objects (no Tk embedding)
  gui/                      Tkinter/customtkinter presentation layer, thin --
                            wires widgets/events to core/* functions.
    thread_bridge.py         Thread-safe callback handoff (see its docstring --
                            this is the fix for a real Tcl crash that background
                            Cellpose threads were triggering)
    box_sync.py               Background worker watching a Box-synced folder
    main_window.py             The main application window
    analytics_window.py        Pop-up fit/pooled-comparison dashboard
tests/                      pytest suite for core/* (fast, no GUI/GPU required)
```

## Tests

```
py -3.12 -m pytest tests/ -v
```

Covers the pieces most likely to break silently on a future edit: the
fitting math, the two real regex bugs found and fixed this session (see
`test_metadata_parsing.py`'s docstrings), cell classification logic, the
intensity-trace pipeline, and replicate pooling. None of it requires a
display, a GPU, or Cellpose weights -- it's pure numpy/pandas.

## Why `gc.disable()`

See the docstring at the top of `gui/thread_bridge.py`. Short version:
matplotlib Figures have reference cycles, Python's automatic garbage
collector can run its cleanup pass on whichever thread happens to be
allocating heavily (the Cellpose/torch background thread, during
auto-segmentation), and if that pass finalizes a Tk-linked object from a
non-main thread, Tcl crashes the whole process. Don't remove `gc.disable()`
from `main.py` without also removing every explicit `gc.collect()` call
that replaces it, or expect that crash back.
