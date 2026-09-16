import os
import re
import time
import gc
import shutil
import pickle
import threading
import queue
import numpy as np
import pandas as pd
import tkinter as tk
import customtkinter as ctk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
from skimage.measure import find_contours

# Prevent Matplotlib memory warning when generating dynamic plots in GUI loops
matplotlib.rcParams['figure.max_open_warning'] = 0

# Matplotlib Figures/Tk canvases hold reference cycles (Figure<->Axes<->Canvas),
# so closing one doesn't free it via simple refcounting -- it needs a cyclic GC
# pass. CPython's automatic GC can run that pass on *whichever thread* happens
# to trip the allocation threshold, which during auto-segmentation is the
# Cellpose/torch worker thread. If that pass finalizes a Tk-linked object
# (e.g. the canvas's internal PhotoImage) from a background thread, Tcl fatally
# crashes with "Tcl_AsyncDelete: async handler deleted by the wrong thread".
# Disabling automatic GC and only ever collecting explicitly on the main
# thread (see build_interactive_canvas) avoids that entirely.
gc.disable()

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==============================================================================
# MATHEMATICAL FIT ENGINE
# ==============================================================================
def exponential_decay(t, A, k, C):
    return A * np.exp(-k * t) + C

def fit_decay_constant(time_axis, intensity_profile):
    try:
        if len(time_axis) < 4 or np.any(np.isnan(intensity_profile)):
            return np.nan, np.nan, np.nan

        plateau_guess = float(np.min(intensity_profile))
        amplitude_guess = float(intensity_profile[0] - plateau_guess)
        
        quarter_idx = max(1, len(time_axis) // 4)
        delta_t = time_axis[quarter_idx] - time_axis[0]
        delta_y = max(1e-4, intensity_profile[0] - intensity_profile[quarter_idx])
        k_guess = np.clip(delta_y / (delta_t * max(amplitude_guess, 1e-4)), 0.001, 5.0)

        p0 = [amplitude_guess, k_guess, plateau_guess]
        bounds = ([-2.0, 0.0, -1.0], [5.0, 50.0, 3.0])

        popt, _ = curve_fit(exponential_decay, time_axis, intensity_profile, p0=p0, bounds=bounds, maxfev=10000)
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return np.nan, np.nan, np.nan

# ==============================================================================
# FILENAME / METADATA PARSING HELPERS
# ==============================================================================
def detect_condition_from_text(text):
    """Normoxia/Hypoxia detection tolerant of the 'hyoxia' (missing 'p') typo
    that shows up in real filenames — matches any 'hy...ox' spelling."""
    low = text.lower()
    if "norm" in low:
        return "Normoxia"
    if re.search(r'hy\w*ox', low):
        return "Hypoxia"
    return "Unknown"

def detect_power_from_text(text):
    """Find a '40mW' / '40 W' style laser power token. Deliberately avoids a
    trailing \\b: Python's regex treats '_' as a word character, so a token
    immediately followed by an underscore (e.g. '40mW_normoxia') would never
    satisfy \\b and the match would silently fail. A negative lookahead for a
    following letter gets the same "don't swallow the next word" protection
    without that failure mode."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*m?W(?![a-zA-Z])', text, re.IGNORECASE)
    return m.group(1) if m else None

# ==============================================================================
# POP-UP ANALYTICS & GRAPH DASHBOARD WINDOW
# ==============================================================================
class AnalyticsDashboardWindow(ctk.CTkToplevel):
    def __init__(self, master_app):
        super().__init__(master_app)
        self.master_app = master_app

        # Track analytics figures explicitly. Do NOT use plt.close('all') because
        # the main application's LassoSelector is attached to its own Matplotlib figure.
        self.current_fit_fig = None
        self.global_fig = None
        self.power_fig = None

        self.title("Cappel Lab | Spectral Kinetics & Fit Analytics")
        self.geometry("960x740")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.tabs_frame = ctk.CTkTabview(self)
        self.tabs_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_current = self.tabs_frame.add("Current Fit Profiles")
        self.tab_global = self.tabs_frame.add("Global Comparative Summary")
        self.tab_power = self.tabs_frame.add("Power Response")

        self.selected_wavelength_var = tk.StringVar(value="")
        self.power_plot_frame = None

        self.refresh_views()

    def refresh_views(self):
        self.render_current_fits()
        self.render_global_summary()
        self.render_power_response()
        # Reclaim the figures just closed above (see gc.disable() note near the
        # top of this file) -- this always runs on the main/Tk thread.
        gc.collect()

    def render_current_fits(self):
        # Close only the previous analytics figure. Never close all Matplotlib figures.
        if self.current_fit_fig is not None:
            try:
                plt.close(self.current_fit_fig)
            except Exception:
                pass
            self.current_fit_fig = None

        for widget in self.tab_current.winfo_children():
            widget.destroy()

        name = self.master_app.dataset_name
        if name not in self.master_app.active_run_traces_cache:
            lbl = ctk.CTkLabel(self.tab_current, text="No active decay profile recorded for this run.", font=ctk.CTkFont(slant="italic"))
            lbl.pack(expand=True)
            return

        traces = self.master_app.active_run_traces_cache[name]
        time_axis = traces["time_axis"]
        gfp_t_corr = traces["gfp_data"]
        A_gfp, k_gfp, C_gfp = traces["gfp_fit"]
        mch_t_corr = traces["mch_data"]
        A_mch, k_mcherry, C_mch = traces["mch_fit"]

        fig_fit, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4.5), facecolor="#2B2B2B")
        ax1.set_facecolor("#1E1E1E"); ax2.set_facecolor("#1E1E1E")

        ax1.plot(time_axis, gfp_t_corr, 'go', alpha=0.4, label="Data")
        if not np.isnan(k_gfp): 
            ax1.plot(time_axis, exponential_decay(time_axis, A_gfp, k_gfp, C_gfp), 'g-', label=f"k={k_gfp:.4f} frame⁻¹")
        if traces["has_gfp_control"]: 
            c_lbl = "Control (Borrowed)" if traces.get("borrowed_control") else "Control (Bleach)"
            ax1.plot(time_axis, traces["gfp_control"], 'g--', label=c_lbl)
        ax1.set_title("GFP-Lamin A Decays", color="white", fontsize=10)
        ax1.set_xlabel("Frames", color="white"); ax1.tick_params(colors="white")
        ax1.legend(labelcolor="white", facecolor="#2B2B2B", edgecolor="none")

        ax2.plot(time_axis, mch_t_corr, 'ro', alpha=0.4, label="Data")
        if not np.isnan(k_mcherry): 
            ax2.plot(time_axis, exponential_decay(time_axis, A_mch, k_mcherry, C_mch), 'r-', label=f"k={k_mcherry:.4f} frame⁻¹")
        if traces["has_mch_control"]: 
            c_lbl = "Control (Borrowed)" if traces.get("borrowed_control") else "Control (Bleach)"
            ax2.plot(time_axis, traces["mch_control"], 'r--', label=c_lbl)
        ax2.set_title("mCherry Decays", color="white", fontsize=10)
        ax2.set_xlabel("Frames", color="white"); ax2.tick_params(colors="white")
        ax2.legend(labelcolor="white", facecolor="#2B2B2B", edgecolor="none")

        fig_fit.tight_layout()
        self.current_fit_fig = fig_fit
        canvas_fit = FigureCanvasTkAgg(fig_fit, master=self.tab_current)
        canvas_fit.get_tk_widget().pack(fill="both", expand=True)
        canvas_fit.draw()

    def render_global_summary(self):
        # Close only the previous analytics figure. The main image/Lasso figure must survive.
        if self.global_fig is not None:
            try:
                plt.close(self.global_fig)
            except Exception:
                pass
            self.global_fig = None

        for widget in self.tab_global.winfo_children():
            widget.destroy()

        fig_global = self.master_app.generate_plots_engine(dark_mode=True)
        if fig_global is None:
            lbl = ctk.CTkLabel(self.tab_global, text="No pooled replicate data found in summary CSV.", font=ctk.CTkFont(slant="italic"))
            lbl.pack(expand=True)
            return

        self.global_fig = fig_global
        canvas_global = FigureCanvasTkAgg(fig_global, master=self.tab_global)
        canvas_global.get_tk_widget().pack(fill="both", expand=True)
        canvas_global.draw()

    def render_power_response(self):
        # Close only the previous power-response figure.
        if self.power_fig is not None:
            try:
                plt.close(self.power_fig)
            except Exception:
                pass
            self.power_fig = None

        for widget in self.tab_power.winfo_children():
            widget.destroy()

        wavelengths = self.master_app.get_available_wavelengths()
        if not wavelengths:
            lbl = ctk.CTkLabel(
                self.tab_power,
                text="No pooled data with a recorded laser power found yet.\n"
                     "Enter a power (mW) in the sidebar before saving a run.",
                font=ctk.CTkFont(slant="italic")
            )
            lbl.pack(expand=True)
            return

        control_bar = ctk.CTkFrame(self.tab_power, fg_color="transparent")
        control_bar.pack(pady=(6, 0), padx=10, fill="x")
        lbl = ctk.CTkLabel(control_bar, text="Wavelength:", font=ctk.CTkFont(weight="bold"))
        lbl.pack(side="left", padx=(0, 6))

        wl_strings = [f"{w:g}nm" for w in wavelengths]
        if self.selected_wavelength_var.get() not in wl_strings:
            self.selected_wavelength_var.set(wl_strings[0])

        dropdown = ctk.CTkOptionMenu(
            control_bar, values=wl_strings, variable=self.selected_wavelength_var,
            command=lambda _: self._draw_power_response_plot()
        )
        dropdown.pack(side="left")

        self.power_plot_frame = ctk.CTkFrame(self.tab_power, fg_color="transparent")
        self.power_plot_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self._draw_power_response_plot()

    def _draw_power_response_plot(self):
        if self.power_fig is not None:
            try:
                plt.close(self.power_fig)
            except Exception:
                pass
            self.power_fig = None

        for widget in self.power_plot_frame.winfo_children():
            widget.destroy()

        wl_str = self.selected_wavelength_var.get()
        if not wl_str:
            return
        try:
            wl_val = float(re.sub(r'[^\d.]', '', wl_str))
        except ValueError:
            return

        fig = self.master_app.generate_power_response_plot(wl_val, dark_mode=True)
        if fig is None:
            lbl = ctk.CTkLabel(
                self.power_plot_frame,
                text=f"No power-tagged replicates recorded at {wl_str} yet.",
                font=ctk.CTkFont(slant="italic")
            )
            lbl.pack(expand=True)
            return

        self.power_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=self.power_plot_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        gc.collect()

    def on_close(self):
        for fig_attr in ("current_fit_fig", "global_fig", "power_fig"):
            fig = getattr(self, fig_attr, None)
            if fig is not None:
                try:
                    plt.close(fig)
                except Exception:
                    pass
                setattr(self, fig_attr, None)
        gc.collect()

        self.master_app.analytics_window = None
        self.destroy()

# ==============================================================================
# MAIN BATCH ANALYZER APPLICATION
# ==============================================================================
class AdvancedBatchCellAnalyzer(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Cappel Lab | Laser Kinetics & Action Spectroscopy Dashboard")
        self.geometry("1220x940")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.fig_canvas = None
        self.tk_canvas = None
        self.im_artist = None
        self.lasso = None
        self.analytics_window = None
        
        self.dataset_prefixes = []
        self.current_dataset_index = -1
        self.master_folder_path = ""
        self.box_source_path = ""
        
        self.gfp_stack = None
        self.mcherry_stack = None
        self.mask_frame = None
        self.dataset_name = "None"
        self.wavelength = "Unknown"
        self.condition = "Unknown"
        self.power_mw = "Unknown"
        self.x_pixels = 400
        self.y_pixels = 400
        self.num_frames = 50
        
        self.curr_vmin = 0.0
        self.curr_vmax = 1.0
        
        self.masks = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.patches = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.raw_hand_masks = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.scatter_artists = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        
        # Auto-segmentation (Cellpose) state. Model is loaded lazily on first
        # use and cached for the life of the app so repeated runs don't pay
        # the weight-load cost again.
        self.cellpose_model = None
        self.auto_seg_running = False
        self.auto_seg_thread = None

        # Fully-automatic batch mode: chains auto-segment -> pixel-count gate ->
        # save-or-skip -> next dataset, across the whole queue, unattended.
        self.auto_run_active = False
        self.auto_run_skip_log = []

        self.active_run_traces_cache = {}
        # Stores shared control traces keyed by (Condition, Wavelength, Power) tuple.
        # Power is included because photobleaching rate is itself power-dependent,
        # so a control captured at one power should not silently be reused at another.
        self.shared_control_library = {}
        
        # Threading / Ingestion Tracker
        self.sync_active = False
        self.sync_thread = None
        self.sync_lock = threading.Lock()
        self.last_sync_log = "Idle"
        self.count_downloaded = 0
        self.count_pending = 0
        # prefix -> human-readable reason it's stuck pending (missing a channel
        # file, or present but still zero bytes / mid-upload), surfaced in the
        # metadata box so a naming mismatch doesn't fail silently in the background.
        self.pending_reasons = {}
        self.last_anomaly_text = ""

        # Worker threads must never touch Tk/Tcl directly (not even via
        # self.after) -- that's the "Tcl_AsyncDelete: async handler deleted by
        # the wrong thread" crash. They hand callbacks to this thread-safe
        # queue instead; only the main-thread poller below ever calls them.
        self.main_thread_queue = queue.Queue()

        self.setup_ui_layout()
        self.after(50, self._poll_main_thread_queue)

    def post_to_main_thread(self, fn):
        """Thread-safe: call from any background thread to run fn() on the main/Tk thread."""
        self.main_thread_queue.put(fn)

    def _poll_main_thread_queue(self):
        while True:
            try:
                fn = self.main_thread_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception as e:
                print(f"Error in queued main-thread callback: {e}")
        self.after(50, self._poll_main_thread_queue)

    def setup_ui_layout(self):
        self.grid_columnconfigure(0, weight=0, minsize=360)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # PANEL 1: SIDEBAR (scrollable, since the control stack no longer fits one screen)
        self.sidebar = ctk.CTkScrollableFrame(self, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        lbl_title = ctk.CTkLabel(self.sidebar, text="Cappel Lab Controls", font=ctk.CTkFont(size=16, weight="bold"))
        lbl_title.pack(pady=(8, 4), padx=20)

        self.btn_load_master = ctk.CTkButton(self.sidebar, text="1. Select Working Folder", command=self.load_flat_data_folder, fg_color="#1F6AA5")
        self.btn_load_master.pack(pady=2, padx=20, fill="x")

        self.btn_sync_box = ctk.CTkButton(self.sidebar, text="2. Stream From Box Folder", command=self.setup_box_background_sync, fg_color="#2E7D32", hover_color="#1B5E20")
        self.btn_sync_box.pack(pady=2, padx=20, fill="x")

        self.btn_new_session = ctk.CTkButton(self.sidebar, text="🆕 Start New Session (Archive Old Results)", command=self.start_new_session, fg_color="#B9770E", hover_color="#9C640C")
        self.btn_new_session.pack(pady=(2, 2), padx=20, fill="x")

        # Ingestion Tracker Box
        self.sync_status_frame = ctk.CTkFrame(self.sidebar, fg_color="#181818", corner_radius=6)
        self.sync_status_frame.pack(pady=(4, 6), padx=20, fill="x")

        self.lbl_download_counter = ctk.CTkLabel(
            self.sync_status_frame, 
            text="Downloaded: 0  |  Pending: 0", 
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#A9DFBF"
        )
        self.lbl_download_counter.pack(pady=3, padx=10)

        self.lbl_queue_progress = ctk.CTkLabel(self.sidebar, text="Queue: 0/0 Completed", font=ctk.CTkFont(size=12, slant="italic"))
        self.lbl_queue_progress.pack(pady=(2, 2), padx=20, anchor="w")
        self.progress_bar = ctk.CTkProgressBar(self.sidebar)
        self.progress_bar.pack(pady=(0, 4), padx=20, fill="x")
        self.progress_bar.set(0)

        self.config_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.config_frame.pack(pady=2, padx=20, fill="x")
        
        self.lbl_frames_entry = ctk.CTkLabel(self.config_frame, text="Default Frames:", font=ctk.CTkFont(weight="bold"))
        self.lbl_frames_entry.grid(row=0, column=0, sticky="w")
        self.frames_entry = ctk.CTkEntry(self.config_frame, width=60)
        self.frames_entry.grid(row=0, column=1, sticky="e")
        self.frames_entry.insert(0, "50")

        self.seg_condition = ctk.CTkSegmentedButton(self.sidebar, values=["Auto (Detect)", "Normoxia", "Hypoxia"], command=self.on_condition_selector_changed)
        self.seg_condition.set("Auto (Detect)")
        self.seg_condition.pack(pady=3, padx=20, fill="x")

        # Laser Power Entry (auto-detected from filename/parameter file, but editable)
        self.power_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.power_frame.pack(pady=2, padx=20, fill="x")
        self.power_frame.grid_columnconfigure(0, weight=1)

        self.lbl_power_entry = ctk.CTkLabel(self.power_frame, text="Laser Power (mW):", font=ctk.CTkFont(weight="bold"))
        self.lbl_power_entry.grid(row=0, column=0, sticky="w")
        self.power_entry = ctk.CTkEntry(self.power_frame, width=60)
        self.power_entry.grid(row=0, column=1, sticky="e")
        self.power_entry.insert(0, "Unknown")
        self.power_entry.bind("<Return>", self.on_power_entry_changed)
        self.power_entry.bind("<FocusOut>", self.on_power_entry_changed)

        self.seg_channel = ctk.CTkSegmentedButton(self.sidebar, values=["GFP (Lamin A) View", "mCherry (Chromatin) View"], command=self.on_channel_view_changed)
        self.seg_channel.set("GFP (Lamin A) View")
        self.seg_channel.pack(pady=3, padx=20, fill="x")

        # Display Contrast Frame
        self.contrast_frame = ctk.CTkFrame(self.sidebar, fg_color="#1E1E1E", corner_radius=6)
        self.contrast_frame.pack(pady=4, padx=15, fill="x")

        lbl_contrast = ctk.CTkLabel(self.contrast_frame, text="Display Contrast (Non-Destructive)", font=ctk.CTkFont(size=11, weight="bold"))
        lbl_contrast.pack(pady=(4, 2), padx=10, anchor="w")

        self.btn_auto_contrast = ctk.CTkButton(self.contrast_frame, text="⚡ Auto Contrast", height=24, fg_color="#34495E", hover_color="#2C3E50", command=self.apply_auto_contrast)
        self.btn_auto_contrast.pack(pady=(0, 4), padx=10, fill="x")

        self.lbl_cmin = ctk.CTkLabel(self.contrast_frame, text="Black Point: 0.00", font=ctk.CTkFont(size=10))
        self.lbl_cmin.pack(pady=(0, 0), padx=10, anchor="w")
        self.slider_cmin = ctk.CTkSlider(self.contrast_frame, from_=0, to=1, number_of_steps=100, command=self.on_contrast_slider_changed)
        self.slider_cmin.pack(pady=(0, 2), padx=10, fill="x")
        self.slider_cmin.set(0.0)

        self.lbl_cmax = ctk.CTkLabel(self.contrast_frame, text="White Point: 1.00", font=ctk.CTkFont(size=10))
        self.lbl_cmax.pack(pady=(0, 0), padx=10, anchor="w")
        self.slider_cmax = ctk.CTkSlider(self.contrast_frame, from_=0, to=1, number_of_steps=100, command=self.on_contrast_slider_changed)
        self.slider_cmax.pack(pady=(0, 4), padx=10, fill="x")
        self.slider_cmax.set(1.0)

        # Local Threshold
        self.lbl_slider_title = ctk.CTkLabel(self.sidebar, text="Local Threshold: 20%", font=ctk.CTkFont(weight="bold"))
        self.lbl_slider_title.pack(pady=(4, 0), padx=20, anchor="w")
        self.thresh_slider = ctk.CTkSlider(self.sidebar, from_=0, to=95, number_of_steps=19, command=self.on_thresh_slider_changed)
        self.thresh_slider.pack(pady=(0, 4), padx=20, fill="x")
        self.thresh_slider.set(20)

        self.phase_var = tk.StringVar(value="TREATED")
        self.rdo_treated = ctk.CTkRadioButton(self.sidebar, text="Treated Cell (Red)", variable=self.phase_var, value="TREATED")
        self.rdo_treated.pack(pady=1, padx=30, anchor="w")
        self.rdo_control = ctk.CTkRadioButton(self.sidebar, text="Control Cell (Blue)", variable=self.phase_var, value="CONTROL")
        self.rdo_control.pack(pady=1, padx=30, anchor="w")
        self.rdo_bg = ctk.CTkRadioButton(self.sidebar, text="Background (Cyan)", variable=self.phase_var, value="BACKGROUND")
        self.rdo_bg.pack(pady=1, padx=30, anchor="w")

        # Shared Control Reuse Checkbox
        self.chk_use_shared_control = ctk.CTkCheckBox(
            self.sidebar, 
            text="Reuse Shared Control (Matching Cond+Wave+Power)", 
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self.on_toggle_shared_control
        )
        self.chk_use_shared_control.pack(pady=(6, 2), padx=20, anchor="w")

        # Auto-Segmentation (Cellpose) Panel
        self.autoseg_frame = ctk.CTkFrame(self.sidebar, fg_color="#1E1E1E", corner_radius=6)
        self.autoseg_frame.pack(pady=(4, 2), padx=15, fill="x")

        lbl_autoseg = ctk.CTkLabel(self.autoseg_frame, text="Auto-Segmentation (Cellpose)", font=ctk.CTkFont(size=11, weight="bold"))
        lbl_autoseg.pack(pady=(4, 2), padx=10, anchor="w")

        self.seg_source_var = tk.StringVar(value="mCherry (Chromatin)")
        self.seg_source_menu = ctk.CTkOptionMenu(self.autoseg_frame, values=["mCherry (Chromatin)", "GFP (Lamin A)"], variable=self.seg_source_var)
        self.seg_source_menu.pack(pady=(0, 4), padx=10, fill="x")

        self.autoseg_params_frame = ctk.CTkFrame(self.autoseg_frame, fg_color="transparent")
        self.autoseg_params_frame.pack(pady=0, padx=10, fill="x")
        self.autoseg_params_frame.grid_columnconfigure(0, weight=1)

        lbl_diam = ctk.CTkLabel(self.autoseg_params_frame, text="Cell Diameter (px, blank=auto):", font=ctk.CTkFont(size=10))
        lbl_diam.grid(row=0, column=0, sticky="w")
        self.entry_cell_diameter = ctk.CTkEntry(self.autoseg_params_frame, width=45)
        self.entry_cell_diameter.grid(row=0, column=1, sticky="e")

        lbl_overlap = ctk.CTkLabel(self.autoseg_params_frame, text="Treated Overlap %:", font=ctk.CTkFont(size=10))
        lbl_overlap.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.entry_overlap_thresh = ctk.CTkEntry(self.autoseg_params_frame, width=45)
        self.entry_overlap_thresh.grid(row=1, column=1, sticky="e", pady=(2, 0))
        self.entry_overlap_thresh.insert(0, "30")

        lbl_minarea = ctk.CTkLabel(self.autoseg_params_frame, text="Min Cell Area (px):", font=ctk.CTkFont(size=10))
        lbl_minarea.grid(row=2, column=0, sticky="w", pady=(2, 0))
        self.entry_min_area = ctk.CTkEntry(self.autoseg_params_frame, width=45)
        self.entry_min_area.grid(row=2, column=1, sticky="e", pady=(2, 0))
        self.entry_min_area.insert(0, "40")

        self.chk_auto_control = ctk.CTkCheckBox(
            self.autoseg_frame,
            text="Auto-Classify Control Cells",
            font=ctk.CTkFont(size=10, weight="bold")
        )
        self.chk_auto_control.pack(pady=(4, 2), padx=10, anchor="w")
        self.chk_auto_control.select()

        self.btn_auto_segment = ctk.CTkButton(self.autoseg_frame, text="🤖 Auto-Segment Cells", fg_color="#16A085", hover_color="#138D75", command=self.run_auto_segmentation)
        self.btn_auto_segment.pack(pady=(6, 4), padx=10, fill="x")

        self.lbl_auto_seg_status = ctk.CTkLabel(self.autoseg_frame, text="Idle", font=ctk.CTkFont(size=10, slant="italic"), wraplength=280, justify="left")
        self.lbl_auto_seg_status.pack(pady=(0, 6), padx=10, anchor="w")

        # Fully-Automatic Batch Mode Panel
        self.autorun_frame = ctk.CTkFrame(self.sidebar, fg_color="#1E1E1E", corner_radius=6)
        self.autorun_frame.pack(pady=(2, 4), padx=15, fill="x")

        lbl_autorun = ctk.CTkLabel(self.autorun_frame, text="Fully Automatic Batch Mode", font=ctk.CTkFont(size=11, weight="bold"))
        lbl_autorun.pack(pady=(4, 2), padx=10, anchor="w")

        self.autorun_params_frame = ctk.CTkFrame(self.autorun_frame, fg_color="transparent")
        self.autorun_params_frame.pack(pady=0, padx=10, fill="x")
        self.autorun_params_frame.grid_columnconfigure(0, weight=1)

        lbl_min_t = ctk.CTkLabel(self.autorun_params_frame, text="Min Treated Px:", font=ctk.CTkFont(size=10))
        lbl_min_t.grid(row=0, column=0, sticky="w")
        self.entry_min_treated_px = ctk.CTkEntry(self.autorun_params_frame, width=45)
        self.entry_min_treated_px.grid(row=0, column=1, sticky="e")
        self.entry_min_treated_px.insert(0, "100")

        lbl_min_c = ctk.CTkLabel(self.autorun_params_frame, text="Min Control Px:", font=ctk.CTkFont(size=10))
        lbl_min_c.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.entry_min_control_px = ctk.CTkEntry(self.autorun_params_frame, width=45)
        self.entry_min_control_px.grid(row=1, column=1, sticky="e", pady=(2, 0))
        self.entry_min_control_px.insert(0, "100")

        lbl_min_b = ctk.CTkLabel(self.autorun_params_frame, text="Min Background Px:", font=ctk.CTkFont(size=10))
        lbl_min_b.grid(row=2, column=0, sticky="w", pady=(2, 0))
        self.entry_min_bg_px = ctk.CTkEntry(self.autorun_params_frame, width=45)
        self.entry_min_bg_px.grid(row=2, column=1, sticky="e", pady=(2, 0))
        self.entry_min_bg_px.insert(0, "100")

        lbl_autorun_note = ctk.CTkLabel(
            self.autorun_frame,
            text="Min Control Px is ignored when Auto-Classify Control Cells is off.",
            font=ctk.CTkFont(size=9, slant="italic"), text_color="#999999",
            wraplength=280, justify="left"
        )
        lbl_autorun_note.pack(pady=(2, 4), padx=10, anchor="w")

        self.btn_auto_run_queue = ctk.CTkButton(self.autorun_frame, text="▶️ Auto-Run Entire Queue", font=ctk.CTkFont(weight="bold"), fg_color="#16A085", hover_color="#138D75", command=self.toggle_auto_run_queue)
        self.btn_auto_run_queue.pack(pady=(2, 4), padx=10, fill="x")

        self.lbl_auto_run_status = ctk.CTkLabel(self.autorun_frame, text="Idle", font=ctk.CTkFont(size=10, slant="italic"), wraplength=280, justify="left")
        self.lbl_auto_run_status.pack(pady=(0, 6), padx=10, anchor="w")

        self.btn_clear_current = ctk.CTkButton(self.sidebar, text="Clear Traces", fg_color="#7F8C8D", hover_color="#95A5A6", command=self.clear_current_traces)
        self.btn_clear_current.pack(pady=(4, 2), padx=20, fill="x")

        self.btn_undo = ctk.CTkButton(self.sidebar, text="Undo Shape", fg_color="#D9534F", hover_color="#C9302C", command=self.undo_last_shape)
        self.btn_undo.pack(pady=2, padx=20, fill="x")

        # Navigation row: Prev / Skip
        self.nav_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.nav_frame.pack(pady=2, padx=20, fill="x")
        self.nav_frame.grid_columnconfigure(0, weight=1)
        self.nav_frame.grid_columnconfigure(1, weight=1)

        self.btn_prev = ctk.CTkButton(self.nav_frame, text="← Prev", fg_color="#E67E22", hover_color="#D35400", command=self.navigate_to_previous_set)
        self.btn_prev.grid(row=0, column=0, sticky="ew", padx=(0, 2))

        self.btn_skip = ctk.CTkButton(self.nav_frame, text="⏭ Skip", fg_color="#6C7A89", hover_color="#596573", command=self.skip_current_set)
        self.btn_skip.grid(row=0, column=1, sticky="ew", padx=(2, 0))

        self.btn_process = ctk.CTkButton(self.sidebar, text="Save & Next →", font=ctk.CTkFont(weight="bold"), fg_color="#5CB85C", hover_color="#4CAE4C", command=self.process_and_advance_queue)
        self.btn_process.pack(pady=2, padx=20, fill="x")

        # Pop-Up Graph Window Button
        self.btn_open_analytics = ctk.CTkButton(self.sidebar, text="📊 Open Analytics Window", font=ctk.CTkFont(weight="bold"), fg_color="#2980B9", hover_color="#3498DB", command=self.toggle_analytics_window)
        self.btn_open_analytics.pack(pady=(6, 2), padx=20, fill="x")

        self.btn_export_plots = ctk.CTkButton(self.sidebar, text="💾 Export Charts", font=ctk.CTkFont(weight="bold"), fg_color="#8E44AD", hover_color="#7D3C98", command=self.export_publication_plots)
        self.btn_export_plots.pack(pady=2, padx=20, fill="x")

        self.metadata_box = ctk.CTkTextbox(self.sidebar, height=100, activate_scrollbars=True, font=ctk.CTkFont(family="monospace", size=10))
        self.metadata_box.pack(pady=(4, 4), padx=20, fill="x")
        self.metadata_box.insert("0.0", "Status: Idle\nBox Stream: Offline")
        self.metadata_box.configure(state="disabled")

        # PANEL 2: MAIN IMAGE CANVAS
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.lbl_canvas_status = ctk.CTkLabel(self.canvas_frame, text="Select working directory or stream from Box to begin.", font=ctk.CTkFont(slant="italic"))
        self.lbl_canvas_status.pack(expand=True)

    # ==============================================================================
    # DISK PERSISTENCE (SAVE & LOAD CACHE & SHARED CONTROLS)
    # ==============================================================================
    def save_cache_to_disk(self):
        if not self.master_folder_path: return
        cache_path = os.path.join(self.master_folder_path, "session_traces_cache.pkl")
        lib_path = os.path.join(self.master_folder_path, "shared_control_library.pkl")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self.active_run_traces_cache, f)
            with open(lib_path, "wb") as f:
                pickle.dump(self.shared_control_library, f)
        except Exception as e:
            print(f"Error saving session cache/library: {e}")

    def load_cache_from_disk(self):
        if not self.master_folder_path: return
        cache_path = os.path.join(self.master_folder_path, "session_traces_cache.pkl")
        lib_path = os.path.join(self.master_folder_path, "shared_control_library.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.active_run_traces_cache = pickle.load(f)
            except Exception as e:
                print(f"Error loading session cache: {e}")
        if os.path.exists(lib_path):
            try:
                with open(lib_path, "rb") as f:
                    self.shared_control_library = pickle.load(f)
            except Exception as e:
                print(f"Error loading control library: {e}")

    def start_new_session(self):
        """Archive the current working folder's summary CSV/cache/shared-control
        library out of the way and reset in-memory state, so newly saved runs
        (and the pooled plots built from them) start clean instead of blending
        with whatever was already processed in this folder."""
        if not self.master_folder_path:
            tk.messagebox.showwarning("No Folder", "Select a working folder first.")
            return

        session_files = ["master_kinetics_summary.csv", "session_traces_cache.pkl", "shared_control_library.pkl"]
        existing = [f for f in session_files if os.path.exists(os.path.join(self.master_folder_path, f))]

        if existing:
            answer = tk.messagebox.askyesno(
                "Start New Session",
                "This moves the existing summary CSV, trace cache, and shared-control "
                "library into an 'Archived_Sessions' subfolder (your raw image data files "
                "are untouched) and starts a fresh summary. Continue?"
            )
            if not answer:
                return

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_dir = os.path.join(self.master_folder_path, "Archived_Sessions", timestamp)
            os.makedirs(archive_dir, exist_ok=True)
            for f in existing:
                shutil.move(os.path.join(self.master_folder_path, f), os.path.join(archive_dir, f))

        self.active_run_traces_cache = {}
        self.shared_control_library = {}
        if self.gfp_stack is not None:
            self.clear_current_traces()
        self.chk_use_shared_control.deselect()
        self.rdo_control.configure(state="normal")

        self.update_analytics_window_if_open()
        if self.current_dataset_index >= 0:
            self.update_metadata_box_display()

        if existing:
            tk.messagebox.showinfo("New Session Started", f"Archived {len(existing)} file(s) to:\n{archive_dir}\n\nStarting fresh.")
        else:
            tk.messagebox.showinfo("New Session Started", "No prior summary/cache found in this folder — already starting fresh.")

    def on_toggle_shared_control(self):
        key = (self.condition, self.wavelength, self.power_mw)
        if self.chk_use_shared_control.get() == 1:
            if key not in self.shared_control_library:
                tk.messagebox.showinfo(
                    "Shared Control Unavailable",
                    f"No baseline control recorded yet for Condition: '{self.condition}', "
                    f"Wavelength: '{self.wavelength}', Power: '{self.power_mw}mW'.\n\n"
                    "Select a Control Cell (Blue) on a reference image first to register one into the shared library."
                )
                self.chk_use_shared_control.deselect()
            else:
                self.rdo_control.configure(state="disabled")
        else:
            self.rdo_control.configure(state="normal")
        self.update_metadata_box_display()

    # ==============================================================================
    # POP-UP / MODAL GRAPH MANAGEMENT
    # ==============================================================================
    def toggle_analytics_window(self):
        if self.analytics_window is None or not self.analytics_window.winfo_exists():
            self.analytics_window = AnalyticsDashboardWindow(self)
        else:
            self.analytics_window.lift()
            self.analytics_window.refresh_views()

    def update_analytics_window_if_open(self):
        if self.analytics_window is not None and self.analytics_window.winfo_exists():
            self.analytics_window.refresh_views()

    # ==============================================================================
    # CONTRAST MANAGEMENT (NON-DESTRUCTIVE)
    # ==============================================================================
    def get_active_display_frame(self):
        if self.seg_channel.get() == "mCherry (Chromatin) View":
            return self.mcherry_stack[0] if self.mcherry_stack is not None else None
        else:
            return self.gfp_stack[0] if self.gfp_stack is not None else None

    def setup_contrast_slider_limits(self, frame):
        if frame is None: return
        f_min = float(np.min(frame))
        f_max = float(np.max(frame))
        if f_max <= f_min: f_max = f_min + 1.0

        self.slider_cmin.configure(from_=f_min, to=f_max)
        self.slider_cmax.configure(from_=f_min, to=f_max)
        self.apply_auto_contrast()

    def apply_auto_contrast(self):
        frame = self.get_active_display_frame()
        if frame is None: return
        
        p_low = float(np.percentile(frame, 1))
        p_high = float(np.percentile(frame, 99.5))
        if p_high <= p_low: p_high = p_low + 1.0

        self.curr_vmin = p_low
        self.curr_vmax = p_high

        self.slider_cmin.set(p_low)
        self.slider_cmax.set(p_high)
        self.lbl_cmin.configure(text=f"Black Point: {p_low:.1f}")
        self.lbl_cmax.configure(text=f"White Point: {p_high:.1f}")

        if self.im_artist is not None:
            self.im_artist.set_clim(self.curr_vmin, self.curr_vmax)
            self.tk_canvas.draw_idle()

    def on_contrast_slider_changed(self, _):
        c_min = self.slider_cmin.get()
        c_max = self.slider_cmax.get()
        
        if c_min >= c_max:
            c_max = c_min + 0.1
            self.slider_cmax.set(c_max)

        self.curr_vmin = c_min
        self.curr_vmax = c_max
        self.lbl_cmin.configure(text=f"Black Point: {c_min:.1f}")
        self.lbl_cmax.configure(text=f"White Point: {c_max:.1f}")

        if self.im_artist is not None:
            self.im_artist.set_clim(self.curr_vmin, self.curr_vmax)
            self.tk_canvas.draw_idle()

    # ==============================================================================
    # BOX BACKGROUND WORKER & DYNAMIC QUEUEING
    # ==============================================================================
    def setup_box_background_sync(self):
        if not self.master_folder_path:
            tk.messagebox.showwarning("Prerequisite", "Select a local destination working folder first using button 1.")
            return

        box_dir = tk.filedialog.askdirectory(title="Select Source Box Drive / Sync Folder")
        if not box_dir:
            return

        self.box_source_path = box_dir
        self.sync_active = True
        self.sync_thread = threading.Thread(target=self._background_box_worker, daemon=True)
        self.sync_thread.start()
        self.btn_sync_box.configure(text="● Box Streaming Active", fg_color="#1B5E20")
        self.last_sync_log = f"Listening on: {os.path.basename(self.box_source_path)}"
        self.update_metadata_box_display()

    def _background_box_worker(self):
        while self.sync_active:
            try:
                if os.path.exists(self.box_source_path) and os.path.exists(self.master_folder_path):
                    box_files = os.listdir(self.box_source_path)
                    
                    all_box_prefixes = set()
                    for f in box_files:
                        if f.lower().endswith(".txt"):
                            p = re.split(r'_(GFP|mCherry|LaminA|Chromatin|Mask|Parameter|Parameters)', f, flags=re.IGNORECASE)[0]
                            all_box_prefixes.add(p)
                            
                    pending_count = 0

                    for prefix in all_box_prefixes:
                        gfp_src = next((f for f in box_files if f.lower() in [f"{prefix.lower()}_gfp.txt", f"{prefix.lower()}_lamina.txt"]), None)
                        mch_src = next((f for f in box_files if f.lower() in [f"{prefix.lower()}_mcherry.txt", f"{prefix.lower()}_chromatin.txt"]), None)
                        msk_src = next((f for f in box_files if f.lower() == f"{prefix.lower()}_mask.txt"), None)
                        
                        if gfp_src and mch_src and msk_src:
                            files_to_copy = [gfp_src, mch_src, msk_src]
                            param_src = next((f for f in box_files if f.lower() in [f"{prefix.lower()}_parameters.txt", f"{prefix.lower()}_parameter.txt"]), None)
                            if param_src:
                                files_to_copy.append(param_src)

                            all_ready = all(os.path.getsize(os.path.join(self.box_source_path, f)) > 0 for f in files_to_copy)

                            if all_ready:
                                copied_any = False
                                for f in files_to_copy:
                                    src = os.path.join(self.box_source_path, f)
                                    dst = os.path.join(self.master_folder_path, f)
                                    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                                        shutil.copy2(src, dst)
                                        copied_any = True

                                with self.sync_lock:
                                    self.pending_reasons.pop(prefix, None)
                                    if prefix not in self.dataset_prefixes:
                                        self.dataset_prefixes.append(prefix)
                                        self.dataset_prefixes.sort()
                                        self.last_sync_log = f"Synced: {prefix}"

                                        if self.current_dataset_index == -1:
                                            self.current_dataset_index = 0
                                            self.post_to_main_thread(lambda idx=0: self.initialize_dataset_at_index(idx))
                                        else:
                                            self.post_to_main_thread(self._update_queue_ui_only)
                                    elif copied_any:
                                        self.last_sync_log = f"Updated files for: {prefix}"
                            else:
                                pending_count += 1
                                self.pending_reasons[prefix] = "files found but still 0 bytes (mid-upload?)"
                        else:
                            pending_count += 1
                            missing = []
                            if not gfp_src: missing.append("GFP/LaminA")
                            if not mch_src: missing.append("mCherry/Chromatin")
                            if not msk_src: missing.append("Mask")
                            self.pending_reasons[prefix] = f"missing {', '.join(missing)} file"

                    self.count_downloaded = len(self.dataset_prefixes)
                    self.count_pending = pending_count
                    self.post_to_main_thread(self._update_sync_counter_labels)
                    self.post_to_main_thread(self.update_metadata_box_display)
            except Exception as e:
                self.last_sync_log = f"Sync error: {str(e)[:25]}"
            time.sleep(3.0)

    def _update_sync_counter_labels(self):
        self.lbl_download_counter.configure(
            text=f"Downloaded: {self.count_downloaded}  |  Pending: {self.count_pending}"
        )

    def _update_queue_ui_only(self):
        total_sets = len(self.dataset_prefixes)
        self.lbl_queue_progress.configure(text=f"Batch: {self.current_dataset_index + 1}/{total_sets}")
        self.progress_bar.set((self.current_dataset_index) / max(1, total_sets))
        self.update_metadata_box_display()

    # ==============================================================================
    # CANVAS & LASSO THRESHOLDING
    # ==============================================================================
    def on_condition_selector_changed(self, value):
        if 0 <= self.current_dataset_index < len(self.dataset_prefixes):
            prefix = self.dataset_prefixes[self.current_dataset_index]
            if value == "Auto (Detect)":
                self.condition = detect_condition_from_text(prefix)
            else:
                self.condition = value
            self.update_metadata_box_display()

    def on_power_entry_changed(self, event=None):
        val = self.power_entry.get().strip()
        self.power_mw = val if val else "Unknown"

        # The shared-control key depends on power, so re-validate whether the
        # currently selected shared control still applies.
        key = (self.condition, self.wavelength, self.power_mw)
        if self.chk_use_shared_control.get() == 1 and key not in self.shared_control_library:
            self.chk_use_shared_control.deselect()
            self.rdo_control.configure(state="normal")
            tk.messagebox.showinfo(
                "Shared Control Unavailable",
                f"No baseline control recorded yet at Power: '{self.power_mw}mW' for this "
                f"Condition/Wavelength. Shared-control reuse has been turned off for this run."
            )
        self.update_metadata_box_display()

    def on_channel_view_changed(self, _):
        frame = self.get_active_display_frame()
        self.setup_contrast_slider_limits(frame)
        self.build_interactive_canvas()
        for phase in ["TREATED", "CONTROL", "BACKGROUND"]:
            for idx in range(len(self.raw_hand_masks[phase])):
                self.apply_threshold_to_slot(phase, idx)
        self.tk_canvas.draw_idle()

    def on_thresh_slider_changed(self, value):
        pct = int(value)
        self.lbl_slider_title.configure(text=f"Local Threshold: {pct}%")
        phase = self.phase_var.get()
        for idx in range(len(self.raw_hand_masks[phase])):
            self.apply_threshold_to_slot(phase, idx)
        self.tk_canvas.draw_idle()

    def apply_threshold_to_slot(self, phase, index):
        raw_mask = self.raw_hand_masks[phase][index]
        color = 'r' if phase == "TREATED" else 'b' if phase == "CONTROL" else 'c'
        
        if self.scatter_artists[phase][index] is not None:
            self.scatter_artists[phase][index].remove()
            self.scatter_artists[phase][index] = None

        final_mask = raw_mask
        
        if phase in ["TREATED", "CONTROL"]:
            active_frame = self.get_active_display_frame()
            local_pixels = active_frame[raw_mask]
            if local_pixels.size > 0:
                mn, mx = np.min(local_pixels), np.max(local_pixels)
                thresh_line = mn + (self.thresh_slider.get() / 100.0) * (mx - mn)
                candidate_mask = np.logical_and(raw_mask, active_frame >= thresh_line)
                if np.sum(candidate_mask) > 0:
                    final_mask = candidate_mask

        self.masks[phase][index] = final_mask

        y_idx, x_idx = np.where(final_mask)
        if y_idx.size > 0:
            scatter_art = self.ax_canvas.scatter(x_idx, y_idx, color=color, s=2, alpha=0.45)
            self.scatter_artists[phase][index] = scatter_art

    def load_flat_data_folder(self):
        self.master_folder_path = tk.filedialog.askdirectory(title="Select Local Working Directory")
        if not self.master_folder_path: return
        
        self.load_cache_from_disk()

        all_files = os.listdir(self.master_folder_path)
        prefixes = set()
        for f in all_files:
            if f.lower().endswith(".txt"):
                base_prefix = re.split(r'_(GFP|mCherry|LaminA|Chromatin|Mask|Parameter|Parameters)', f, flags=re.IGNORECASE)[0]
                prefixes.add(base_prefix)
                
        valid_prefixes = [p for p in prefixes if any(f.lower() in [f"{p.lower()}_gfp.txt", f"{p.lower()}_lamina.txt"] for f in all_files)
                          and any(f.lower() in [f"{p.lower()}_mcherry.txt", f"{p.lower()}_chromatin.txt"] for f in all_files)
                          and any(f.lower() == f"{p.lower()}_mask.txt" for f in all_files)]
                
        with self.sync_lock:
            self.dataset_prefixes = sorted(valid_prefixes)
            self.count_downloaded = len(self.dataset_prefixes)
            self._update_sync_counter_labels()
            
        if self.dataset_prefixes:
            self.current_dataset_index = 0
            self.initialize_dataset_at_index(self.current_dataset_index)
        else:
            self.lbl_canvas_status.configure(text="Local folder loaded (0 sets). Ready to stream from Box.")

    def initialize_dataset_at_index(self, index):
        if index < 0 or index >= len(self.dataset_prefixes): return
        prefix = self.dataset_prefixes[index]
        self.dataset_name = prefix
        
        all_files = os.listdir(self.master_folder_path)
        gfp_file = next(f for f in all_files if f.lower() in [f"{prefix.lower()}_gfp.txt", f"{prefix.lower()}_lamina.txt"])
        mcherry_file = next(f for f in all_files if f.lower() in [f"{prefix.lower()}_mcherry.txt", f"{prefix.lower()}_chromatin.txt"])
        mask_file = next(f for f in all_files if f.lower() == f"{prefix.lower()}_mask.txt")
        param_file = next((f for f in all_files if f.lower() in [f"{prefix.lower()}_parameters.txt", f"{prefix.lower()}_parameter.txt"]), None)

        self.x_pixels = 400
        self.y_pixels = 400
        detected_power = None

        if param_file:
            try:
                with open(os.path.join(self.master_folder_path, param_file), 'r') as f:
                    content = f.read()
                x_match = re.search(r'# of Pixels:\s*(\d+)', content)
                if x_match:
                    self.x_pixels = int(x_match.group(1))
                    self.y_pixels = self.x_pixels
                # Look for a "Power" field in the parameter file as a fallback
                # source (used only if the filename itself doesn't state it).
                p_match = re.search(r'Power[^\d\-]*(\d+(?:\.\d+)?)', content, re.IGNORECASE)
                if p_match:
                    detected_power = p_match.group(1)
                else:
                    # No explicit "Power" label — fall back to a bare "40mW" style token anywhere in the file.
                    detected_power = detect_power_from_text(content)
            except Exception:
                pass

        wave_match = re.search(r'(\d+)\s*nm', prefix, re.IGNORECASE) or re.search(r'(\d{3,4})', prefix)
        self.wavelength = f"{wave_match.group(1)}nm" if wave_match else "Unknown"

        # Laser power: prefer an explicit "...50mW..." style token in the filename
        # prefix over anything pulled from the parameter file.
        power_from_name = detect_power_from_text(prefix)
        if power_from_name:
            detected_power = power_from_name
        self.power_mw = detected_power if detected_power else "Unknown"
        self.power_entry.delete(0, tk.END)
        self.power_entry.insert(0, self.power_mw)

        selected_cond = self.seg_condition.get()
        if selected_cond == "Auto (Detect)":
            self.condition = detect_condition_from_text(prefix)
        else:
            self.condition = selected_cond

        def parse_data_file(name):
            with open(os.path.join(self.master_folder_path, name), 'r') as f:
                content = f.read()
            return np.array(re.findall(r'[-+]?\d*\.\d+|\d+', content), dtype=np.float64)

        gfp_data = parse_data_file(gfp_file)
        mcherry_data = parse_data_file(mcherry_file)
        mask_data = parse_data_file(mask_file)

        pts_per_frame = self.x_pixels * self.y_pixels
        
        calculated_frames = len(gfp_data) // pts_per_frame
        if calculated_frames < 1:
            tk.messagebox.showerror("Data Error", f"File {gfp_file} has insufficient points ({len(gfp_data)}) for {self.x_pixels}x{self.y_pixels} resolution.")
            return

        self.num_frames = calculated_frames
        self.frames_entry.delete(0, tk.END)
        self.frames_entry.insert(0, str(self.num_frames))

        total_pts_to_use = self.num_frames * pts_per_frame
        self.gfp_stack = gfp_data[:total_pts_to_use].reshape((self.num_frames, self.y_pixels, self.x_pixels))
        self.mcherry_stack = mcherry_data[:total_pts_to_use].reshape((self.num_frames, self.y_pixels, self.x_pixels))
        self.mask_frame = mask_data[:pts_per_frame].reshape((self.y_pixels, self.x_pixels))

        total_sets = len(self.dataset_prefixes)
        self.lbl_queue_progress.configure(text=f"Batch: {index + 1}/{total_sets}")
        self.progress_bar.set(index / max(1, total_sets))

        # Check if a shared control is available for this Condition + Wavelength + Power combination
        key = (self.condition, self.wavelength, self.power_mw)
        if key in self.shared_control_library:
            self.chk_use_shared_control.select()
            self.rdo_control.configure(state="disabled")
        else:
            self.chk_use_shared_control.deselect()
            self.rdo_control.configure(state="normal")

        self.setup_contrast_slider_limits(self.get_active_display_frame())
        self.update_metadata_box_display()
        self.clear_current_traces()

    def update_metadata_box_display(self, anomaly_text=None):
        # anomaly_text=None means "leave whatever spread-warning was showing
        # alone" (this is re-invoked every few seconds by the background sync,
        # which has nothing to say about that warning either way).
        if anomaly_text is not None:
            self.last_anomaly_text = anomaly_text
        anomaly_text = getattr(self, "last_anomaly_text", "")

        total_sets = len(self.dataset_prefixes)
        stream_status = "Streaming" if self.sync_active else "Offline"

        key = (self.condition, self.wavelength, self.power_mw)
        has_lib_control = key in self.shared_control_library
        control_mode_str = "Shared Available" if has_lib_control else "Needs Local Selection"
        if self.chk_use_shared_control.get() == 1:
            control_mode_str = "Reusing Shared Control"

        self.metadata_box.configure(state="normal")
        self.metadata_box.delete("0.0", "end")
        msg = (
            f"Run: [{self.current_dataset_index+1}/{total_sets}]\n"
            f"Prefix: {self.dataset_name}\n"
            f"Wave: {self.wavelength} | Cond: {self.condition} | Power: {self.power_mw}mW\n"
            f"Control Mode: {control_mode_str}\n"
            f"Dim: {self.x_pixels}x{self.y_pixels} | Frames: {self.num_frames}\n"
            f"Box: {stream_status} ({self.last_sync_log})"
        )
        if self.count_pending > 0 and self.pending_reasons:
            shown = list(self.pending_reasons.items())[:3]
            lines = [f"  • {p}: {r}" for p, r in shown]
            extra = len(self.pending_reasons) - len(shown)
            if extra > 0:
                lines.append(f"  • ...and {extra} more")
            msg += f"\n\n{self.count_pending} Pending:\n" + "\n".join(lines)
        if anomaly_text:
            msg += f"\n\n⚠️ SPREAD WARNING:\n{anomaly_text}"
        self.metadata_box.insert("0.0", msg)
        self.metadata_box.configure(state="disabled")

    def clear_current_traces(self):
        self.masks = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.patches = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.raw_hand_masks = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.scatter_artists = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.phase_var.set("TREATED")
        self.build_interactive_canvas()

    def navigate_to_previous_set(self):
        if self.current_dataset_index > 0:
            self.current_dataset_index -= 1
            self.initialize_dataset_at_index(self.current_dataset_index)
            self.update_analytics_window_if_open()

    def skip_current_set(self):
        if self.current_dataset_index < 0: return
        self.current_dataset_index += 1
        if self.current_dataset_index < len(self.dataset_prefixes):
            self.initialize_dataset_at_index(self.current_dataset_index)
            self.update_analytics_window_if_open()
        else:
            self.progress_bar.set(1.0)
            self.lbl_queue_progress.configure(text=f"Batch Complete ({len(self.dataset_prefixes)} sets) 🎉")
            tk.messagebox.showinfo("Done", "Caught up to the end of the current queue.")

    def build_interactive_canvas(self):
        self.update_idletasks()
        
        # Properly disconnect old Lasso and destroy only the previous main canvas.
        # This is intentionally targeted; never use plt.close('all') here.
        if self.lasso is not None:
            try:
                self.lasso.disconnect_events()
            except Exception:
                pass
            self.lasso = None

        if self.fig_canvas is not None:
            try:
                plt.close(self.fig_canvas.figure)
            except Exception:
                pass
            self.fig_canvas = None

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        # Reclaim the just-closed figure's reference cycles now, deterministically,
        # on the main thread -- see the gc.disable() note near the top of this file.
        gc.collect()

        self.fig_canvas, self.ax_canvas = plt.subplots(figsize=(6.5, 6.5), facecolor="#2B2B2B")
        
        if self.seg_channel.get() == "mCherry (Chromatin) View":
            self.im_artist = self.ax_canvas.imshow(self.mcherry_stack[0], cmap='inferno', vmin=self.curr_vmin, vmax=self.curr_vmax)
        else:
            self.im_artist = self.ax_canvas.imshow(self.gfp_stack[0], cmap='gray', vmin=self.curr_vmin, vmax=self.curr_vmax)
            display_mask = np.where(self.mask_frame > (np.max(self.mask_frame) * 0.1), 1, 0)
            self.ax_canvas.imshow(display_mask, cmap='Reds', alpha=0.15)
            
        self.ax_canvas.axis('off')
        self.fig_canvas.tight_layout()
        
        self.tk_canvas = FigureCanvasTkAgg(self.fig_canvas, master=self.canvas_frame)
        self.tk_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Clean re-binding of LassoSelector to active axes canvas
        self.lasso = LassoSelector(self.ax_canvas, self.on_lasso_release, props=dict(color='yellow', linewidth=1.5))
        self.tk_canvas.draw()

    def on_lasso_release(self, verts):
        if len(verts) < 3: return
        phase = self.phase_var.get()
        color = 'r' if phase == "TREATED" else 'b' if phase == "CONTROL" else 'c'
        
        y_grid, x_grid = np.mgrid[:self.y_pixels, :self.x_pixels]
        points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
        raw_mask = Path(verts).contains_points(points).reshape((self.y_pixels, self.x_pixels))
        
        poly = Polygon(verts, edgecolor=color, facecolor='none', linewidth=1.2, linestyle=':')
        self.ax_canvas.add_patch(poly)
        
        self.raw_hand_masks[phase].append(raw_mask)
        self.patches[phase].append(poly)
        self.masks[phase].append(raw_mask)
        self.scatter_artists[phase].append(None)
        
        active_idx = len(self.raw_hand_masks[phase]) - 1
        self.apply_threshold_to_slot(phase, active_idx)
        self.tk_canvas.draw_idle()

    def undo_last_shape(self):
        phase = self.phase_var.get()
        if self.raw_hand_masks[phase]:
            self.raw_hand_masks[phase].pop()
            self.masks[phase].pop()
            self.patches[phase].pop().remove()
            if self.scatter_artists[phase][-1] is not None:
                self.scatter_artists[phase].pop().remove()
            else:
                self.scatter_artists[phase].pop()
            self.tk_canvas.draw_idle()

    # ==============================================================================
    # AUTO-SEGMENTATION (CELLPOSE)
    # ==============================================================================
    def update_auto_seg_status(self, text):
        self.lbl_auto_seg_status.configure(text=text)

    def get_cellpose_model(self):
        if self.cellpose_model is not None:
            return self.cellpose_model
        from cellpose import models
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False
        self.cellpose_model = models.CellposeModel(gpu=use_gpu, pretrained_model='cyto3')
        return self.cellpose_model

    def run_auto_segmentation(self, on_complete=None):
        if self.gfp_stack is None or self.mcherry_stack is None:
            tk.messagebox.showwarning("No Data", "Load a dataset before running auto-segmentation.")
            return
        if self.auto_seg_running:
            return

        seg_choice = self.seg_source_var.get()
        source_frame = self.mcherry_stack[0] if "mCherry" in seg_choice else self.gfp_stack[0]
        laser_ref_frame = self.mask_frame
        dataset_snapshot = self.dataset_name

        diam_raw = self.entry_cell_diameter.get().strip()
        try:
            diameter = float(diam_raw) if diam_raw else None
        except ValueError:
            diameter = None

        self.auto_seg_running = True
        self.btn_auto_segment.configure(state="disabled", text="⏳ Segmenting...")
        self.update_auto_seg_status(f"Running Cellpose on {seg_choice}...")

        def worker():
            try:
                model = self.get_cellpose_model()
                img = np.asarray(source_frame, dtype=np.float32)
                labels, _, _ = model.eval(img, channels=[0, 0], diameter=diameter)
                self.post_to_main_thread(lambda: self._on_auto_segmentation_done(labels, laser_ref_frame, dataset_snapshot, on_complete))
            except Exception as e:
                err_msg = str(e)
                self.post_to_main_thread(lambda: self._on_auto_segmentation_failed(err_msg, on_complete))

        self.auto_seg_thread = threading.Thread(target=worker, daemon=True)
        self.auto_seg_thread.start()

    def _on_auto_segmentation_failed(self, err_msg, on_complete=None):
        self.auto_seg_running = False
        self.btn_auto_segment.configure(state="normal", text="🤖 Auto-Segment Cells")
        self.update_auto_seg_status(f"Segmentation failed: {err_msg[:120]}")
        if not self.auto_run_active:
            tk.messagebox.showerror("Auto-Segment Failed", f"Cellpose segmentation failed:\n{err_msg}")
        if on_complete:
            on_complete()

    def _on_auto_segmentation_done(self, labels, laser_ref_frame, dataset_snapshot, on_complete=None):
        self.auto_seg_running = False
        self.btn_auto_segment.configure(state="normal", text="🤖 Auto-Segment Cells")

        if dataset_snapshot != self.dataset_name:
            self.update_auto_seg_status("Discarded a stale result (you switched datasets mid-run).")
            if on_complete:
                on_complete()
            return

        n_labels = int(labels.max())
        if n_labels == 0:
            self.update_auto_seg_status("No cells detected. Try the other channel, adjust diameter, or select manually.")
            if not self.auto_run_active:
                tk.messagebox.showinfo("Auto-Segment", "Cellpose did not detect any cells in this frame.")
            if on_complete:
                on_complete()
            return

        try:
            min_area = max(1, int(float(self.entry_min_area.get())))
        except ValueError:
            min_area = 40
        try:
            overlap_frac_thresh = np.clip(float(self.entry_overlap_thresh.get()), 0.0, 100.0) / 100.0
        except ValueError:
            overlap_frac_thresh = 0.30

        if laser_ref_frame is not None and np.max(laser_ref_frame) > 0:
            laser_mask = laser_ref_frame > (np.max(laser_ref_frame) * 0.1)
        else:
            laser_mask = np.zeros(labels.shape, dtype=bool)

        self.clear_current_traces()

        auto_control_enabled = self.chk_auto_control.get() == 1

        n_treated, n_control, n_excluded, n_dropped = 0, 0, 0, 0
        all_cells_mask = np.zeros(labels.shape, dtype=bool)

        for label_id in range(1, n_labels + 1):
            cell_mask = labels == label_id
            area = int(np.sum(cell_mask))
            if area < min_area:
                n_dropped += 1
                continue
            # Every detected cell is excluded from the background estimate,
            # regardless of whether it ends up classified as anything below.
            all_cells_mask |= cell_mask
            overlap_frac = np.sum(cell_mask & laser_mask) / area
            if overlap_frac >= overlap_frac_thresh:
                self._add_auto_mask("TREATED", cell_mask)
                n_treated += 1
            elif auto_control_enabled:
                self._add_auto_mask("CONTROL", cell_mask)
                n_control += 1
            else:
                n_excluded += 1

        background_mask = ~all_cells_mask
        if np.any(background_mask):
            self._add_auto_mask("BACKGROUND", background_mask)

        self.tk_canvas.draw_idle()

        summary = f"Detected {n_treated + n_control + n_excluded} cells: {n_treated} treated / {n_control} control"
        if n_excluded:
            summary += f" / {n_excluded} excluded (auto-control off)"
        if n_dropped:
            summary += f" ({n_dropped} dropped, <{min_area}px)"
        if n_treated == 0:
            summary += "\n⚠️ No cell overlapped the laser-target mask — check the overlap threshold or dataset."
        self.update_auto_seg_status(summary)

        if on_complete:
            on_complete()

    def _add_auto_mask(self, phase, mask):
        color = 'r' if phase == "TREATED" else 'b' if phase == "CONTROL" else 'c'

        self.raw_hand_masks[phase].append(mask)
        self.masks[phase].append(mask)

        poly = None
        if phase != "BACKGROUND":
            try:
                contours = find_contours(mask.astype(float), 0.5)
                if contours:
                    longest = max(contours, key=len)
                    verts = np.column_stack((longest[:, 1], longest[:, 0]))
                    poly = Polygon(verts, edgecolor=color, facecolor='none', linewidth=1.0, linestyle='-')
                    self.ax_canvas.add_patch(poly)
            except Exception:
                poly = None
        if poly is None:
            poly = Polygon([[0, 0]], edgecolor=color, facecolor='none', linewidth=0, visible=False)
            self.ax_canvas.add_patch(poly)

        self.patches[phase].append(poly)
        self.scatter_artists[phase].append(None)

        idx = len(self.raw_hand_masks[phase]) - 1
        self.apply_threshold_to_slot(phase, idx)

    # ==============================================================================
    # FITTING & QUEUE PROGRESSION
    # ==============================================================================
    def process_and_advance_queue(self, auto_mode=False):
        if not self.masks["BACKGROUND"] or not self.masks["TREATED"]:
            if not auto_mode:
                tk.messagebox.showwarning("Warning", "Select Treated and Background regions before saving.")
            return False

        key = (self.condition, self.wavelength, self.power_mw)
        using_shared_control = self.chk_use_shared_control.get() == 1

        if not self.masks["CONTROL"] and not using_shared_control:
            if auto_mode:
                # Unattended: proceed without photobleach correction rather than
                # blocking on a confirmation dialog nobody's there to answer.
                pass
            else:
                answer = tk.messagebox.askyesno(
                    "No Control Selected",
                    "No Control cell selected for this dataset, and shared control reuse is OFF.\n\n"
                    "Do you want to proceed fitting without photobleaching correction?"
                )
                if not answer:
                    return False

        def get_mask_average_trace(stack, masks_list):
            intensities = []
            for frame in stack[:self.num_frames]:
                frame_vals = [np.mean(frame[m]) for m in masks_list if np.any(m)]
                intensities.append(np.mean(frame_vals) if frame_vals else 1.0)
            return np.array(intensities)

        gfp_t = get_mask_average_trace(self.gfp_stack, self.masks["TREATED"])
        gfp_bg = get_mask_average_trace(self.gfp_stack, self.masks["BACKGROUND"])

        mch_t = get_mask_average_trace(self.mcherry_stack, self.masks["TREATED"])
        mch_bg = get_mask_average_trace(self.mcherry_stack, self.masks["BACKGROUND"])

        gfp_t_net = np.maximum(gfp_t - gfp_bg, 1e-4)
        mch_t_net = np.maximum(mch_t - mch_bg, 1e-4)

        gfp_t_norm = gfp_t_net / gfp_t_net[0]
        mch_t_norm = mch_t_net / mch_t_net[0]

        # Determine GFP Control Trace
        has_gfp_c = False
        borrowed_c = False
        if self.masks["CONTROL"]:
            gfp_c = get_mask_average_trace(self.gfp_stack, self.masks["CONTROL"])
            gfp_c_net = np.maximum(gfp_c - gfp_bg, 1e-4)
            gfp_c_norm = gfp_c_net / gfp_c_net[0]
            has_gfp_c = True
        elif using_shared_control and key in self.shared_control_library:
            gfp_c_norm = self.shared_control_library[key]["gfp_c_norm"]
            has_gfp_c = True
            borrowed_c = True
        else:
            gfp_c_norm = np.ones(self.num_frames)

        # Determine mCherry Control Trace
        has_mch_c = False
        if self.masks["CONTROL"]:
            mch_c = get_mask_average_trace(self.mcherry_stack, self.masks["CONTROL"])
            mch_c_net = np.maximum(mch_c - mch_bg, 1e-4)
            mch_c_norm = mch_c_net / mch_c_net[0]
            has_mch_c = True
        elif using_shared_control and key in self.shared_control_library:
            mch_c_norm = self.shared_control_library[key]["mch_c_norm"] if "mch_c_norm" in self.shared_control_library[key] else self.shared_control_library[key]["mch_c_norm"]
            has_mch_c = True
            borrowed_c = True
        else:
            mch_c_norm = np.ones(self.num_frames)

        # Update Shared Control Library if fresh local control was drawn
        if self.masks["CONTROL"]:
            self.shared_control_library[key] = {
                "gfp_c_norm": gfp_c_norm,
                "mch_c_norm": mch_c_norm,
                "source_dataset": self.dataset_name
            }

        gfp_t_corr = gfp_t_norm / np.clip(gfp_c_norm, 1e-4, None) if has_gfp_c else gfp_t_norm
        mcherry_t_corr = mch_t_norm / np.clip(mch_c_norm, 1e-4, None) if has_mch_c else mch_t_norm

        time_axis = np.arange(self.num_frames)
        A_gfp, k_gfp, C_gfp = fit_decay_constant(time_axis, gfp_t_corr)
        A_mch, k_mcherry, C_mch = fit_decay_constant(time_axis, mcherry_t_corr)

        self.active_run_traces_cache[self.dataset_name] = {
            "time_axis": time_axis,
            "gfp_data": gfp_t_corr, "gfp_fit": (A_gfp, k_gfp, C_gfp), "has_gfp_control": has_gfp_c, "gfp_control": gfp_c_norm,
            "mch_data": mcherry_t_corr, "mch_fit": (A_mch, k_mcherry, C_mch), "has_mch_control": has_mch_c, "mch_control": mch_c_norm,
            "borrowed_control": borrowed_c
        }

        self.save_cache_to_disk()

        summary_file = os.path.join(self.master_folder_path, "master_kinetics_summary.csv")
        new_rows = [
            {"Dataset": self.dataset_name, "Wavelength": self.wavelength, "Condition": self.condition, "Power_mW": self.power_mw, "Fluorophore": "GFP-LaminA", "Type": "Treated", "Decay_Constant_k": k_gfp},
            {"Dataset": self.dataset_name, "Wavelength": self.wavelength, "Condition": self.condition, "Power_mW": self.power_mw, "Fluorophore": "mCherry", "Type": "Treated", "Decay_Constant_k": k_mcherry}
        ]
        summary_df = pd.DataFrame(new_rows)
        
        try:
            if os.path.exists(summary_file):
                master_df = pd.read_csv(summary_file)
                if "Power_mW" not in master_df.columns:
                    master_df["Power_mW"] = "Unknown"
                master_df = master_df[(master_df["Dataset"] != self.dataset_name)]
                master_df = pd.concat([master_df, summary_df], ignore_index=True)
            else:
                master_df = summary_df
            master_df.to_csv(summary_file, index=False)
        except Exception as e:
            # A save failure (locked file, disk full, etc.) is systemic rather than
            # per-dataset, so this dialog is shown even during auto-run: better to
            # surface it and let the caller halt than silently retry forever.
            tk.messagebox.showerror("Save Error", f"Could not update summary:\n{e}")
            return False

        self.update_analytics_window_if_open()

        self.current_dataset_index += 1
        if self.current_dataset_index < len(self.dataset_prefixes):
            self.initialize_dataset_at_index(self.current_dataset_index)
        else:
            self.progress_bar.set(1.0)
            self.lbl_queue_progress.configure(text=f"Batch Complete ({len(self.dataset_prefixes)} sets) 🎉")
            if not auto_mode:
                tk.messagebox.showinfo("Done", "All currently synced datasets cataloged.")
        return True

    # ==============================================================================
    # FULLY AUTOMATIC BATCH MODE
    # ==============================================================================
    def update_auto_run_status(self, text):
        self.lbl_auto_run_status.configure(text=text)

    def toggle_auto_run_queue(self):
        if self.auto_run_active:
            self.auto_run_active = False
            self.btn_auto_run_queue.configure(text="▶️ Auto-Run Entire Queue", fg_color="#16A085", hover_color="#138D75")
            self.update_auto_run_status("Stopped.")
            return

        if not self.dataset_prefixes:
            tk.messagebox.showwarning("No Data", "Load or sync a dataset queue first.")
            return

        if self.current_dataset_index < 0:
            self.current_dataset_index = 0

        self.auto_run_active = True
        self.auto_run_skip_log = []
        self.btn_auto_run_queue.configure(text="⏹ Stop Auto-Run", fg_color="#C0392B", hover_color="#A93226")
        self._auto_run_process_current()

    def _auto_run_process_current(self):
        if not self.auto_run_active:
            return
        if self.current_dataset_index < 0 or self.current_dataset_index >= len(self.dataset_prefixes):
            self._finish_auto_run()
            return

        self.initialize_dataset_at_index(self.current_dataset_index)
        total = len(self.dataset_prefixes)
        self.update_auto_run_status(f"[{self.current_dataset_index + 1}/{total}] Segmenting {self.dataset_name}...")
        self.run_auto_segmentation(on_complete=self._auto_run_after_segmentation)

    def _auto_run_after_segmentation(self):
        if not self.auto_run_active:
            return

        total = len(self.dataset_prefixes)

        def read_int(entry, default):
            try:
                return max(0, int(float(entry.get())))
            except ValueError:
                return default

        min_treated = read_int(self.entry_min_treated_px, 100)
        min_control = read_int(self.entry_min_control_px, 100)
        min_bg = read_int(self.entry_min_bg_px, 100)

        treated_px = sum(int(np.sum(m)) for m in self.masks["TREATED"])
        control_px = sum(int(np.sum(m)) for m in self.masks["CONTROL"])
        bg_px = sum(int(np.sum(m)) for m in self.masks["BACKGROUND"])
        control_required = self.chk_auto_control.get() == 1

        fail_reasons = []
        if treated_px < min_treated:
            fail_reasons.append(f"treated {treated_px}px < {min_treated}px")
        if control_required and control_px < min_control:
            fail_reasons.append(f"control {control_px}px < {min_control}px")
        if bg_px < min_bg:
            fail_reasons.append(f"background {bg_px}px < {min_bg}px")

        if fail_reasons:
            reason = "; ".join(fail_reasons)
            self.auto_run_skip_log.append((self.dataset_name, reason))
            self.update_auto_run_status(f"[{self.current_dataset_index + 1}/{total}] Skipped {self.dataset_name}: {reason}")
            self._auto_run_advance()
            return

        self.update_auto_run_status(f"[{self.current_dataset_index + 1}/{total}] Saving {self.dataset_name}...")
        saved_ok = self.process_and_advance_queue(auto_mode=True)
        if saved_ok:
            self.after(50, self._auto_run_process_current)
        else:
            self.auto_run_active = False
            self.btn_auto_run_queue.configure(text="▶️ Auto-Run Entire Queue", fg_color="#16A085", hover_color="#138D75")
            self.update_auto_run_status(f"Auto-run stopped: failed to save {self.dataset_name} (see error dialog).")

    def _auto_run_advance(self):
        self.current_dataset_index += 1
        self.after(50, self._auto_run_process_current)

    def _finish_auto_run(self):
        self.auto_run_active = False
        self.btn_auto_run_queue.configure(text="▶️ Auto-Run Entire Queue", fg_color="#16A085", hover_color="#138D75")
        n_skipped = len(self.auto_run_skip_log)
        self.update_auto_run_status(f"Auto-run complete. {n_skipped} skipped.")

        if n_skipped:
            shown = self.auto_run_skip_log[:15]
            lines = "\n".join(f"• {name}: {reason}" for name, reason in shown)
            extra = n_skipped - len(shown)
            if extra > 0:
                lines += f"\n...and {extra} more"
            tk.messagebox.showinfo("Auto-Run Complete", f"Finished the queue. {n_skipped} dataset(s) skipped:\n\n{lines}")
        else:
            tk.messagebox.showinfo("Auto-Run Complete", "Finished processing the entire queue. No datasets were skipped.")

    def process_and_pool_replicates_df(self):
        summary_file = os.path.join(self.master_folder_path, "master_kinetics_summary.csv")
        if not os.path.exists(summary_file) or os.path.getsize(summary_file) < 10: return None, ""
        df = pd.read_csv(summary_file)
        df['Wavelength_Num'] = df['Wavelength'].astype(str).str.extract(r'(\d+)').astype(float)

        if 'Power_mW' in df.columns:
            df['Power_Num'] = df['Power_mW'].astype(str).str.extract(r'(\d+(?:\.\d+)?)').astype(float)
        else:
            df['Power_Num'] = np.nan

        df = df.dropna(subset=['Wavelength_Num', 'Decay_Constant_k'])
        
        # Power is included in the grouping key: a "replicate group" is now defined
        # by matching Wavelength + Power + Condition + Fluorophore, since different
        # laser powers at the same wavelength are expected to give different decay
        # rates and should not be averaged together.
        grouped = df.groupby(["Wavelength_Num", "Power_Num", "Condition", "Fluorophore"], dropna=False).agg(
            Mean_k=('Decay_Constant_k', 'mean'),
            Std_k=('Decay_Constant_k', 'std'),
            Count=('Decay_Constant_k', 'count')
        ).reset_index()
        grouped['Std_k'] = grouped['Std_k'].fillna(0.0)
        
        anomaly_msg = ""
        outliers = grouped[grouped['Std_k'] > 0.05]
        if not outliers.empty:
            lines = []
            for _, r in outliers.iterrows():
                pwr_str = f"{r['Power_Num']:g}mW" if pd.notna(r['Power_Num']) else "power unknown"
                lines.append(f"• {r['Fluorophore']} ({int(r['Wavelength_Num'])}nm, {pwr_str} - {r['Condition']}): High spread (σ={r['Std_k']:.4f}, N={r['Count']})")
            anomaly_msg = "\n".join(lines)
        return grouped, anomaly_msg

    def get_available_wavelengths(self):
        """Wavelengths that have at least one replicate with a known laser power, for the Power Response tab."""
        grouped_df, _ = self.process_and_pool_replicates_df()
        if grouped_df is None or grouped_df.empty:
            return []
        with_power = grouped_df[grouped_df['Power_Num'].notna()]
        return sorted(with_power['Wavelength_Num'].dropna().unique().tolist())

    def generate_plots_engine(self, dark_mode=True):
        grouped_df, anomaly_text = self.process_and_pool_replicates_df()
        if grouped_df is None or grouped_df.empty: return None
        self.update_metadata_box_display(anomaly_text)

        # IMPORTANT: Do not call plt.close('all') here.
        # The main application LassoSelector is attached to its own Matplotlib
        # figure, and closing all figures destroys the active Lasso event canvas.
        # Analytics figures are now tracked and closed individually by
        # AnalyticsDashboardWindow.

        marker_dict = {"Normoxia": "o", "Hypoxia": "s", "Unknown": "^"}
        # Different laser powers at the same wavelength are drawn as the same
        # color (by condition) but a different line style, so power comparisons
        # at a given wavelength are visible directly on these plots too.
        linestyles = ['-', '--', '-.', ':']
        bg_col = "#2B2B2B" if dark_mode else "white"
        face_col = "#1E1E1E" if dark_mode else "#F5F5F5"
        txt_col = "white" if dark_mode else "black"
        grid_col = "gray" if dark_mode else "darkgray"

        fig_global = plt.figure(figsize=(10, 7), facecolor=bg_col)
        gfp_pool = grouped_df[grouped_df["Fluorophore"] == "GFP-LaminA"]
        mch_pool = grouped_df[grouped_df["Fluorophore"] == "mCherry"]

        def power_groups(pool_df):
            powers = sorted(pool_df['Power_Num'].dropna().unique().tolist())
            return powers if powers else [None]

        def power_suffix(pwr):
            return f" @ {pwr:g}mW" if pwr is not None else ""

        def subset_by_power(pool_df, cond, pwr):
            if pwr is None:
                return pool_df[(pool_df["Condition"] == cond) & (pool_df['Power_Num'].isna())].sort_values('Wavelength_Num')
            return pool_df[(pool_df["Condition"] == cond) & (pool_df['Power_Num'] == pwr)].sort_values('Wavelength_Num')

        # 1. GFP Lamin A
        ax1 = plt.subplot2grid((2, 2), (0, 0)); ax1.set_facecolor(face_col)
        if not gfp_pool.empty:
            powers_present = power_groups(gfp_pool)
            for cond, col in [("Normoxia", "#5CB85C"), ("Hypoxia", "#2E7D32")]:
                for p_idx, pwr in enumerate(powers_present):
                    sub = subset_by_power(gfp_pool, cond, pwr)
                    if not sub.empty:
                        ls = linestyles[p_idx % len(linestyles)]
                        ax1.plot(sub['Wavelength_Num'], sub['Mean_k'], color=col, marker=marker_dict.get(cond, "o"), linestyle=ls, markersize=7, linewidth=2, label=f"GFP-LaminA {cond}{power_suffix(pwr)}")
                        ax1.errorbar(sub['Wavelength_Num'], sub['Mean_k'], yerr=sub['Std_k'], fmt='none', ecolor=col, elinewidth=1.2, capsize=3)
        ax1.set_title("GFP-Lamin A Kinetics", color=txt_col, fontsize=11); ax1.tick_params(colors=txt_col); ax1.grid(True, linestyle=":", color=grid_col, alpha=0.3)
        l1 = ax1.legend(facecolor=bg_col, edgecolor="none", fontsize=8); [t.set_color(txt_col) for t in l1.get_texts()] if l1 else None

        # 2. mCherry
        ax2 = plt.subplot2grid((2, 2), (0, 1)); ax2.set_facecolor(face_col)
        if not mch_pool.empty:
            powers_present = power_groups(mch_pool)
            for cond, col in [("Normoxia", "#D9534F"), ("Hypoxia", "#C0392B")]:
                for p_idx, pwr in enumerate(powers_present):
                    sub = subset_by_power(mch_pool, cond, pwr)
                    if not sub.empty:
                        ls = linestyles[p_idx % len(linestyles)]
                        ax2.plot(sub['Wavelength_Num'], sub['Mean_k'], color=col, marker=marker_dict.get(cond, "o"), linestyle=ls, markersize=7, linewidth=2, label=f"mCherry {cond}{power_suffix(pwr)}")
                        ax2.errorbar(sub['Wavelength_Num'], sub['Mean_k'], yerr=sub['Std_k'], fmt='none', ecolor=col, elinewidth=1.2, capsize=3)
        ax2.set_title("mCherry Kinetics", color=txt_col, fontsize=11); ax2.tick_params(colors=txt_col); ax2.grid(True, linestyle=":", color=grid_col, alpha=0.3)
        l2 = ax2.legend(facecolor=bg_col, edgecolor="none", fontsize=8); [t.set_color(txt_col) for t in l2.get_texts()] if l2 else None

        # 3. Unified View
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2); ax3.set_facecolor(face_col)
        color_map = {("GFP-LaminA", "Normoxia"): "#5CB85C", ("GFP-LaminA", "Hypoxia"): "#2E7D32", ("mCherry", "Normoxia"): "#D9534F", ("mCherry", "Hypoxia"): "#C0392B"}
        all_powers_present = power_groups(grouped_df)
        for (fl, cond), grp_all in grouped_df.groupby(["Fluorophore", "Condition"]):
            c = color_map.get((fl, cond), "blue")
            for p_idx, pwr in enumerate(all_powers_present):
                if pwr is None:
                    grp = grp_all[grp_all['Power_Num'].isna()].sort_values('Wavelength_Num')
                else:
                    grp = grp_all[grp_all['Power_Num'] == pwr].sort_values('Wavelength_Num')
                if not grp.empty:
                    ls = linestyles[p_idx % len(linestyles)]
                    lbl = f"LaminA [{cond}]{power_suffix(pwr)}" if "GFP-LaminA" in fl else f"mCherry [{cond}]{power_suffix(pwr)}"
                    ax3.plot(grp['Wavelength_Num'], grp['Mean_k'], color=c, marker=marker_dict.get(cond, "o"), linestyle=ls, markersize=7, linewidth=2, label=lbl)
                    ax3.errorbar(grp['Wavelength_Num'], grp['Mean_k'], yerr=grp['Std_k'], fmt='none', ecolor=c, elinewidth=1.2, capsize=3)
        ax3.set_title("Unified Pooled Response Matrix", color=txt_col, fontsize=11); ax3.set_xlabel("Wavelength (nm)", color=txt_col); ax3.set_ylabel("Mean Decay Rate (k, frame⁻¹)", color=txt_col)
        ax3.tick_params(colors=txt_col); ax3.grid(True, linestyle=":", color=grid_col, alpha=0.3)
        l3 = ax3.legend(facecolor=bg_col, edgecolor="none", fontsize=8); [t.set_color(txt_col) for t in l3.get_texts()] if l3 else None

        fig_global.tight_layout()
        return fig_global

    def generate_power_response_plot(self, wavelength_filter, dark_mode=True):
        """Decay rate (k) vs laser power, at a fixed wavelength, split by fluorophore and condition."""
        grouped_df, _ = self.process_and_pool_replicates_df()
        if grouped_df is None or grouped_df.empty:
            return None

        sub_df = grouped_df[
            np.isclose(grouped_df['Wavelength_Num'], wavelength_filter) & grouped_df['Power_Num'].notna()
        ]
        if sub_df.empty:
            return None

        marker_dict = {"Normoxia": "o", "Hypoxia": "s", "Unknown": "^"}
        bg_col = "#2B2B2B" if dark_mode else "white"
        face_col = "#1E1E1E" if dark_mode else "#F5F5F5"
        txt_col = "white" if dark_mode else "black"
        grid_col = "gray" if dark_mode else "darkgray"
        color_map = {
            ("GFP-LaminA", "Normoxia"): "#5CB85C", ("GFP-LaminA", "Hypoxia"): "#2E7D32",
            ("mCherry", "Normoxia"): "#D9534F", ("mCherry", "Hypoxia"): "#C0392B",
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), facecolor=bg_col)
        ax1.set_facecolor(face_col); ax2.set_facecolor(face_col)

        for fl, ax in [("GFP-LaminA", ax1), ("mCherry", ax2)]:
            fl_df = sub_df[sub_df["Fluorophore"] == fl]
            for cond, grp in fl_df.groupby("Condition"):
                grp = grp.sort_values('Power_Num')
                c = color_map.get((fl, cond), "steelblue")
                ax.plot(grp['Power_Num'], grp['Mean_k'], color=c, marker=marker_dict.get(cond, "o"), markersize=8, linewidth=2, label=cond)
                ax.errorbar(grp['Power_Num'], grp['Mean_k'], yerr=grp['Std_k'], fmt='none', ecolor=c, elinewidth=1.5, capsize=4)
            ax.set_title(f"{fl} — {wavelength_filter:g}nm", color=txt_col, fontsize=11)
            ax.set_xlabel("Laser Power (mW)", color=txt_col)
            ax.set_ylabel("Mean Decay Rate (k, frame⁻¹)", color=txt_col)
            ax.tick_params(colors=txt_col)
            ax.grid(True, linestyle=":", color=grid_col, alpha=0.3)
            leg = ax.legend(facecolor=bg_col, edgecolor="none")
            if leg:
                for t in leg.get_texts():
                    t.set_color(txt_col)

        fig.suptitle(f"Power-Dependence of Decay Kinetics at {wavelength_filter:g}nm", color=txt_col, fontsize=12, weight="bold")
        fig.tight_layout()
        return fig

    def export_publication_plots(self):
        grouped_df, _ = self.process_and_pool_replicates_df()
        if grouped_df is None or grouped_df.empty: return
            
        target_dir = os.path.abspath(self.master_folder_path)
        individual_plots_dir = os.path.join(target_dir, "Individual_Decay_Plots")
        os.makedirs(individual_plots_dir, exist_ok=True)
        
        for name, traces in self.active_run_traces_cache.items():
            fig_ind, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
            t_axis = traces["time_axis"]
            
            ax1.plot(t_axis, traces["gfp_data"], 'go', alpha=0.5, label="Data")
            A, k, C = traces["gfp_fit"]
            if not np.isnan(k): ax1.plot(t_axis, exponential_decay(t_axis, A, k, C), 'g-', label=f"Fit (k={k:.4f} frame⁻¹)")
            if traces["has_gfp_control"]:
                c_lbl = "Control (Borrowed)" if traces.get("borrowed_control") else "Control (Bleach)"
                ax1.plot(t_axis, traces["gfp_control"], 'g--', label=c_lbl)
            ax1.set_title("GFP-Lamin A Profile"); ax1.set_xlabel("Frames"); ax1.set_ylabel("F/F0 (Corrected)"); ax1.grid(True, linestyle=":"); ax1.legend()
            
            ax2.plot(t_axis, traces["mch_data"], 'ro', alpha=0.5, label="Data")
            A_m, k_m, C_m = traces["mch_fit"]
            if not np.isnan(k_m): ax2.plot(t_axis, exponential_decay(t_axis, A_m, k_m, C_m), 'r-', label=f"Fit (k={k_m:.4f} frame⁻¹)")
            if traces["has_mch_control"]:
                c_lbl = "Control (Borrowed)" if traces.get("borrowed_control") else "Control (Bleach)"
                ax2.plot(t_axis, traces["mch_control"], 'r--', label=c_lbl)
            ax2.set_title("mCherry Profile"); ax2.set_xlabel("Frames"); ax2.set_ylabel("F/F0 (Corrected)"); ax2.grid(True, linestyle=":"); ax2.legend()
            
            plt.suptitle(f"Decay Profile: {name}", fontsize=11, weight="bold")
            plt.tight_layout()
            fig_ind.savefig(os.path.join(individual_plots_dir, f"{name}_decay_fit.png"), dpi=300)
            plt.close(fig_ind)

        gfp_pool = grouped_df[grouped_df["Fluorophore"] == "GFP-LaminA"]
        mch_pool = grouped_df[grouped_df["Fluorophore"] == "mCherry"]
        marker_dict = {"Normoxia": "o", "Hypoxia": "s", "Unknown": "^"}
        linestyles = ['-', '--', '-.', ':']

        def power_groups(pool_df):
            powers = sorted(pool_df['Power_Num'].dropna().unique().tolist())
            return powers if powers else [None]

        def power_suffix(pwr):
            return f" @ {pwr:g}mW" if pwr is not None else ""

        def subset_by_power(pool_df, cond, pwr):
            if pwr is None:
                return pool_df[(pool_df["Condition"] == cond) & (pool_df['Power_Num'].isna())].sort_values('Wavelength_Num')
            return pool_df[(pool_df["Condition"] == cond) & (pool_df['Power_Num'] == pwr)].sort_values('Wavelength_Num')

        fig1, ax1 = plt.subplots(figsize=(6, 5))
        powers_present = power_groups(gfp_pool)
        for cond, col in [("Normoxia", "#5CB85C"), ("Hypoxia", "#2E7D32")]:
            for p_idx, pwr in enumerate(powers_present):
                sub = subset_by_power(gfp_pool, cond, pwr)
                if not sub.empty:
                    ls = linestyles[p_idx % len(linestyles)]
                    ax1.plot(sub['Wavelength_Num'], sub['Mean_k'], color=col, marker=marker_dict.get(cond, "o"), linestyle=ls, markersize=8, linewidth=2, label=f"GFP-LaminA {cond}{power_suffix(pwr)}")
                    ax1.errorbar(sub['Wavelength_Num'], sub['Mean_k'], yerr=sub['Std_k'], fmt='none', ecolor=col, elinewidth=1.5, capsize=4)
        ax1.set_title("Pooled GFP-Lamin A Action Decay Spectrum"); ax1.set_xlabel("Wavelength (nm)"); ax1.set_ylabel("Mean k (frame⁻¹)"); ax1.grid(True, linestyle=":"); ax1.legend(fontsize=8)
        fig1.tight_layout(); fig1.savefig(os.path.join(target_dir, "pooled_gfp_laminA_spectral_decay.png"), dpi=300); plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        powers_present = power_groups(mch_pool)
        for cond, col in [("Normoxia", "#D9534F"), ("Hypoxia", "#C0392B")]:
            for p_idx, pwr in enumerate(powers_present):
                sub = subset_by_power(mch_pool, cond, pwr)
                if not sub.empty:
                    ls = linestyles[p_idx % len(linestyles)]
                    ax2.plot(sub['Wavelength_Num'], sub['Mean_k'], color=col, marker=marker_dict.get(cond, "o"), linestyle=ls, markersize=8, linewidth=2, label=f"mCherry {cond}{power_suffix(pwr)}")
                    ax2.errorbar(sub['Wavelength_Num'], sub['Mean_k'], yerr=sub['Std_k'], fmt='none', ecolor=col, elinewidth=1.5, capsize=4)
        ax2.set_title("Pooled mCherry Action Decay Spectrum"); ax2.set_xlabel("Wavelength (nm)"); ax2.set_ylabel("Mean k (frame⁻¹)"); ax2.grid(True, linestyle=":"); ax2.legend(fontsize=8)
        fig2.tight_layout(); fig2.savefig(os.path.join(target_dir, "pooled_mcherry_spectral_decay.png"), dpi=300); plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(8, 5.5))
        color_map = {("GFP-LaminA", "Normoxia"): "#5CB85C", ("GFP-LaminA", "Hypoxia"): "#2E7D32", ("mCherry", "Normoxia"): "#D9534F", ("mCherry", "Hypoxia"): "#C0392B"}
        all_powers_present = power_groups(grouped_df)
        for (fl, cond), grp_all in grouped_df.groupby(["Fluorophore", "Condition"]):
            c = color_map.get((fl, cond), "blue")
            for p_idx, pwr in enumerate(all_powers_present):
                if pwr is None:
                    grp = grp_all[grp_all['Power_Num'].isna()].sort_values('Wavelength_Num')
                else:
                    grp = grp_all[grp_all['Power_Num'] == pwr].sort_values('Wavelength_Num')
                if not grp.empty:
                    ls = linestyles[p_idx % len(linestyles)]
                    lbl = f"LaminA [{cond}]{power_suffix(pwr)}" if "GFP-LaminA" in fl else f"mCherry [{cond}]{power_suffix(pwr)}"
                    ax3.plot(grp['Wavelength_Num'], grp['Mean_k'], color=c, marker=marker_dict.get(cond, "o"), linestyle=ls, markersize=8, linewidth=2, label=lbl)
                    ax3.errorbar(grp['Wavelength_Num'], grp['Mean_k'], yerr=grp['Std_k'], fmt='none', ecolor=c, elinewidth=1.5, capsize=4)
        ax3.set_title("Unified Action Spectroscopy Decay Spectrum"); ax3.set_xlabel("Wavelength (nm)"); ax3.set_ylabel("Mean Decay Rate (k, frame⁻¹)"); ax3.grid(True, linestyle=":"); ax3.legend(fontsize=8)
        fig3.tight_layout(); fig3.savefig(os.path.join(target_dir, "pooled_combined_spectral_kinetics.png"), dpi=300); plt.close(fig3)

        # Per-wavelength power-response plots (k vs Power, for each wavelength that has power-tagged replicates)
        power_plots_dir = os.path.join(target_dir, "Power_Response_Plots")
        wavelengths_with_power = grouped_df[grouped_df['Power_Num'].notna()]['Wavelength_Num'].dropna().unique()
        if len(wavelengths_with_power) > 0:
            os.makedirs(power_plots_dir, exist_ok=True)
            for wl in sorted(wavelengths_with_power):
                fig_p = self.generate_power_response_plot(wl, dark_mode=False)
                if fig_p is not None:
                    fig_p.savefig(os.path.join(power_plots_dir, f"power_response_{wl:g}nm.png"), dpi=300)
                    plt.close(fig_p)

        gc.collect()
        tk.messagebox.showinfo("Export Complete", f"Exported individual decay fits, global response matrices, and per-wavelength power-response plots to:\n{target_dir}")

    def on_closing(self):
        self.sync_active = False
        self.save_cache_to_disk()
        # Safe here because the entire application is shutting down.
        plt.close('all')
        gc.collect()
        self.quit()
        self.destroy()

if __name__ == "__main__":
    app = AdvancedBatchCellAnalyzer()
    app.mainloop()