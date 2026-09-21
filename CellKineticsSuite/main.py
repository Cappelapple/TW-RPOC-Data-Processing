"""Entry point. Run with: py -3.12 main.py

(Python 3.12, not the system default -- see README.md for why: this app
needs GPU-enabled PyTorch for Cellpose to run at interactive speed, and on
a machine with multiple Python installs it's easy to end up launching the
one with only the CPU build.)
"""
import gc

import matplotlib
import customtkinter as ctk

from cell_kinetics.gui.main_window import AdvancedBatchCellAnalyzer

# Matplotlib Figures/Tk canvases hold reference cycles (Figure<->Axes<->Canvas),
# so closing one doesn't free it via simple refcounting -- it needs a cyclic GC
# pass. CPython's automatic GC can run that pass on *whichever thread* happens
# to trip the allocation threshold, which during auto-segmentation is the
# Cellpose/torch worker thread. If that pass finalizes a Tk-linked object
# (e.g. the canvas's internal PhotoImage) from a background thread, Tcl
# fatally crashes with "Tcl_AsyncDelete: async handler deleted by the wrong
# thread". Disabling automatic GC and only ever collecting explicitly on the
# main thread (see gui/main_window.py and gui/analytics_window.py) avoids
# that entirely.
gc.disable()

# Prevent Matplotlib memory warning when generating dynamic plots in GUI loops
matplotlib.rcParams['figure.max_open_warning'] = 0

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

if __name__ == "__main__":
    app = AdvancedBatchCellAnalyzer()
    app.mainloop()
