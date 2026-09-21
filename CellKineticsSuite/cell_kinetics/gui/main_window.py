"""Main application window: sidebar controls, image canvas, and all the
event handlers that wire user actions to core/* logic.

This module intentionally does very little computation itself -- almost
every handler's body is "gather inputs from widgets, call a core function,
put the result on a widget." The actual math/parsing/segmentation/fitting
logic lives in cell_kinetics.core and is unit-tested independently of this
GUI (see tests/).
"""
import os
import gc
import threading

import numpy as np
import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.patches import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..core import fitting, metadata_parsing, dataset_io, segmentation, intensity_traces, persistence, pooling, plotting
from .thread_bridge import MainThreadDispatcher
from .box_sync import BoxSyncWorker
from .analytics_window import AnalyticsDashboardWindow


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

        self.active_run_traces_cache = {}
        # Shared control traces keyed by (Condition, Wavelength, Power). Power is
        # included because photobleaching rate is itself power-dependent, so a
        # control captured at one power should not silently be reused at another.
        self.shared_control_library = {}
        self.last_anomaly_text = ""

        # Threading / Ingestion. dataset_prefixes and sync_lock are shared
        # (same objects) with box_sync, which mutates them in place.
        self.sync_lock = threading.Lock()
        self.dispatcher = MainThreadDispatcher(self)
        self.box_sync = BoxSyncWorker(
            dispatcher=self.dispatcher,
            dataset_prefixes=self.dataset_prefixes,
            sync_lock=self.sync_lock,
            on_dataset_synced=self._on_box_dataset_synced,
            on_status_changed=self._on_box_status_changed,
        )

        # Auto-segmentation (Cellpose). Model is loaded lazily on first use
        # and cached for the life of the app so repeated runs don't pay the
        # weight-load cost again.
        self.cellpose_model = None
        self.auto_seg_running = False
        self.auto_seg_thread = None

        # Fully-automatic batch mode: chains auto-segment -> pixel-count gate
        # -> save-or-skip -> next dataset, across the whole queue, unattended.
        self.auto_run_active = False
        self.auto_run_skip_log = []

        self.setup_ui_layout()
        self.dispatcher.start(50)

    def setup_ui_layout(self):
        self.grid_columnconfigure(0, weight=0, minsize=360)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # PANEL 1: SIDEBAR (scrollable, since the control stack doesn't fit one screen)
        self.sidebar = ctk.CTkScrollableFrame(self, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        lbl_title = ctk.CTkLabel(self.sidebar, text="Cappel Lab Controls", font=ctk.CTkFont(size=16, weight="bold"))
        lbl_title.pack(pady=(8, 4), padx=20)

        self.btn_load_master = ctk.CTkButton(self.sidebar, text="1. Select Working Folder", command=self.load_flat_data_folder, fg_color="#1F6AA5")
        self.btn_load_master.pack(pady=2, padx=20, fill="x")

        self.btn_sync_box = ctk.CTkButton(self.sidebar, text="2. Stream From Box Folder", command=self.setup_box_background_sync, fg_color="#2E7D32", hover_color="#1B5E20")
        self.btn_sync_box.pack(pady=2, padx=20, fill="x")

        self.btn_new_session = ctk.CTkButton(self.sidebar, text="\U0001f195 Start New Session (Archive Old Results)", command=self.start_new_session, fg_color="#B9770E", hover_color="#9C640C")
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

        self.btn_auto_segment = ctk.CTkButton(self.autoseg_frame, text="\U0001f916 Auto-Segment Cells", fg_color="#16A085", hover_color="#138D75", command=self.run_auto_segmentation)
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
        self.btn_open_analytics = ctk.CTkButton(self.sidebar, text="\U0001f4ca Open Analytics Window", font=ctk.CTkFont(weight="bold"), fg_color="#2980B9", hover_color="#3498DB", command=self.toggle_analytics_window)
        self.btn_open_analytics.pack(pady=(6, 2), padx=20, fill="x")

        self.btn_export_plots = ctk.CTkButton(self.sidebar, text="\U0001f4be Export Charts", font=ctk.CTkFont(weight="bold"), fg_color="#8E44AD", hover_color="#7D3C98", command=self.export_publication_plots)
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
        try:
            persistence.save_session_cache(self.master_folder_path, self.active_run_traces_cache, self.shared_control_library)
        except Exception as e:
            print(f"Error saving session cache/library: {e}")

    def load_cache_from_disk(self):
        if not self.master_folder_path: return
        self.active_run_traces_cache, self.shared_control_library = persistence.load_session_cache(self.master_folder_path)

    def start_new_session(self):
        """Archive the current working folder's summary CSV/cache/shared-control
        library out of the way and reset in-memory state, so newly saved runs
        (and the pooled plots built from them) start clean instead of blending
        with whatever was already processed in this folder."""
        if not self.master_folder_path:
            tk.messagebox.showwarning("No Folder", "Select a working folder first.")
            return

        existing_preview = [f for f in persistence.SESSION_FILES if os.path.exists(os.path.join(self.master_folder_path, f))]
        if existing_preview:
            answer = tk.messagebox.askyesno(
                "Start New Session",
                "This moves the existing summary CSV, trace cache, and shared-control "
                "library into an 'Archived_Sessions' subfolder (your raw image data files "
                "are untouched) and starts a fresh summary. Continue?"
            )
            if not answer:
                return

        archived, archive_dir = persistence.archive_session_files(self.master_folder_path)

        self.active_run_traces_cache = {}
        self.shared_control_library = {}
        if self.gfp_stack is not None:
            self.clear_current_traces()
        self.chk_use_shared_control.deselect()
        self.rdo_control.configure(state="normal")

        self.update_analytics_window_if_open()
        if self.current_dataset_index >= 0:
            self.update_metadata_box_display()

        if archived:
            tk.messagebox.showinfo("New Session Started", f"Archived {len(archived)} file(s) to:\n{archive_dir}\n\nStarting fresh.")
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
        self.box_sync.start(box_dir, self.master_folder_path)
        self.btn_sync_box.configure(text="● Box Streaming Active", fg_color="#1B5E20")
        self.update_metadata_box_display()

    def _on_box_dataset_synced(self):
        if self.current_dataset_index == -1 and self.dataset_prefixes:
            self.current_dataset_index = 0
            self.initialize_dataset_at_index(0)
        else:
            self._update_queue_ui_only()

    def _on_box_status_changed(self):
        self._update_sync_counter_labels()
        self.update_metadata_box_display()

    def _update_sync_counter_labels(self):
        self.lbl_download_counter.configure(
            text=f"Downloaded: {self.box_sync.count_downloaded}  |  Pending: {self.box_sync.count_pending}"
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
                self.condition = metadata_parsing.detect_condition_from_text(prefix)
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
            final_mask = segmentation.refine_mask_by_local_threshold(active_frame, raw_mask, self.thresh_slider.get())

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
        discovered = dataset_io.discover_dataset_prefixes(all_files)

        with self.sync_lock:
            # Mutate in place -- box_sync holds this same list object.
            self.dataset_prefixes[:] = discovered
            self.box_sync.count_downloaded = len(self.dataset_prefixes)
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
        channel_files = dataset_io.find_channel_files(prefix, all_files)
        if not dataset_io.is_complete(channel_files):
            tk.messagebox.showerror("Data Error", f"Dataset '{prefix}' is missing required files.")
            return

        self.x_pixels = 400
        self.y_pixels = 400
        detected_power = None

        if channel_files["param"]:
            try:
                with open(os.path.join(self.master_folder_path, channel_files["param"]), 'r') as f:
                    content = f.read()
                params = metadata_parsing.parse_parameters_file(content)
                if "x_pixels" in params:
                    self.x_pixels = params["x_pixels"]
                    self.y_pixels = self.x_pixels
                if "power" in params:
                    detected_power = params["power"]
            except Exception:
                pass

        wave = metadata_parsing.detect_wavelength_from_text(prefix)
        self.wavelength = f"{wave}nm" if wave else "Unknown"

        # Laser power: prefer an explicit "...50mW..." style token in the
        # filename prefix over anything pulled from the parameter file.
        power_from_name = metadata_parsing.detect_power_from_text(prefix)
        if power_from_name:
            detected_power = power_from_name
        self.power_mw = detected_power if detected_power else "Unknown"
        self.power_entry.delete(0, tk.END)
        self.power_entry.insert(0, self.power_mw)

        selected_cond = self.seg_condition.get()
        if selected_cond == "Auto (Detect)":
            self.condition = metadata_parsing.detect_condition_from_text(prefix)
        else:
            self.condition = selected_cond

        gfp_path = os.path.join(self.master_folder_path, channel_files["gfp"])
        mcherry_path = os.path.join(self.master_folder_path, channel_files["mcherry"])
        mask_path = os.path.join(self.master_folder_path, channel_files["mask"])

        try:
            self.gfp_stack, self.mcherry_stack, self.mask_frame, self.num_frames = dataset_io.load_dataset_stacks(
                gfp_path, mcherry_path, mask_path, self.x_pixels, self.y_pixels
            )
        except ValueError as e:
            tk.messagebox.showerror("Data Error", str(e))
            return

        self.frames_entry.delete(0, tk.END)
        self.frames_entry.insert(0, str(self.num_frames))

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
        anomaly_text = self.last_anomaly_text

        total_sets = len(self.dataset_prefixes)
        stream_status = "Streaming" if self.box_sync.active else "Offline"

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
            f"Box: {stream_status} ({self.box_sync.last_sync_log})"
        )
        if self.box_sync.count_pending > 0 and self.box_sync.pending_reasons:
            shown = list(self.box_sync.pending_reasons.items())[:3]
            lines = [f"  • {p}: {r}" for p, r in shown]
            extra = len(self.box_sync.pending_reasons) - len(shown)
            if extra > 0:
                lines.append(f"  • ...and {extra} more")
            msg += f"\n\n{self.box_sync.count_pending} Pending:\n" + "\n".join(lines)
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
            self.lbl_queue_progress.configure(text=f"Batch Complete ({len(self.dataset_prefixes)} sets) \U0001f389")
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
        # on the main thread -- see thread_bridge.py's docstring for why this
        # matters when a Cellpose background thread might be running.
        gc.collect()

        self.fig_canvas, self.ax_canvas = plt.subplots(figsize=(6.5, 6.5), facecolor="#2B2B2B")

        if self.seg_channel.get() == "mCherry (Chromatin) View":
            self.im_artist = self.ax_canvas.imshow(self.mcherry_stack[0], cmap='inferno', vmin=self.curr_vmin, vmax=self.curr_vmax)
        else:
            self.im_artist = self.ax_canvas.imshow(self.gfp_stack[0], cmap='gray', vmin=self.curr_vmin, vmax=self.curr_vmax)
            display_mask = segmentation.build_laser_mask(self.mask_frame, self.mask_frame.shape).astype(int)
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

        raw_mask = segmentation.polygon_to_mask(verts, self.y_pixels, self.x_pixels)

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
        self.cellpose_model = segmentation.load_cellpose_model()
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
                labels = segmentation.run_segmentation(model, source_frame, diameter=diameter)
                self.dispatcher.post(lambda: self._on_auto_segmentation_done(labels, laser_ref_frame, dataset_snapshot, on_complete))
            except Exception as e:
                err_msg = str(e)
                self.dispatcher.post(lambda: self._on_auto_segmentation_failed(err_msg, on_complete))

        self.auto_seg_thread = threading.Thread(target=worker, daemon=True)
        self.auto_seg_thread.start()

    def _on_auto_segmentation_failed(self, err_msg, on_complete=None):
        self.auto_seg_running = False
        self.btn_auto_segment.configure(state="normal", text="\U0001f916 Auto-Segment Cells")
        self.update_auto_seg_status(f"Segmentation failed: {err_msg[:120]}")
        if not self.auto_run_active:
            tk.messagebox.showerror("Auto-Segment Failed", f"Cellpose segmentation failed:\n{err_msg}")
        if on_complete:
            on_complete()

    def _on_auto_segmentation_done(self, labels, laser_ref_frame, dataset_snapshot, on_complete=None):
        self.auto_seg_running = False
        self.btn_auto_segment.configure(state="normal", text="\U0001f916 Auto-Segment Cells")

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

        laser_mask = segmentation.build_laser_mask(laser_ref_frame, labels.shape)
        auto_control_enabled = self.chk_auto_control.get() == 1

        self.clear_current_traces()

        result = segmentation.classify_cells(labels, laser_mask, min_area, overlap_frac_thresh, auto_control_enabled)

        for cell_mask in result["treated_masks"]:
            self._add_auto_mask("TREATED", cell_mask)
        for cell_mask in result["control_masks"]:
            self._add_auto_mask("CONTROL", cell_mask)
        if np.any(result["background_mask"]):
            self._add_auto_mask("BACKGROUND", result["background_mask"])

        self.tk_canvas.draw_idle()

        n_treated, n_control, n_excluded, n_dropped = (
            result["n_treated"], result["n_control"], result["n_excluded"], result["n_dropped"]
        )
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
            verts = segmentation.mask_to_contour_vertices(mask)
            if verts is not None:
                poly = Polygon(verts, edgecolor=color, facecolor='none', linewidth=1.0, linestyle='-')
                self.ax_canvas.add_patch(poly)
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

        fallback_gfp = fallback_mch = None
        if not self.masks["CONTROL"] and using_shared_control and key in self.shared_control_library:
            fallback_gfp = self.shared_control_library[key]["gfp_c_norm"]
            fallback_mch = self.shared_control_library[key]["mch_c_norm"]

        trace_result = intensity_traces.compute_corrected_traces(
            self.gfp_stack, self.mcherry_stack,
            self.masks["TREATED"], self.masks["CONTROL"], self.masks["BACKGROUND"],
            self.num_frames,
            fallback_gfp_control_norm=fallback_gfp,
            fallback_mch_control_norm=fallback_mch,
        )

        # Update Shared Control Library if a fresh local control was drawn
        if self.masks["CONTROL"]:
            self.shared_control_library[key] = {
                "gfp_c_norm": trace_result["gfp_control_norm"],
                "mch_c_norm": trace_result["mch_control_norm"],
                "source_dataset": self.dataset_name,
            }

        time_axis = np.arange(self.num_frames)
        A_gfp, k_gfp, C_gfp = fitting.fit_decay_constant(time_axis, trace_result["gfp_t_corr"])
        A_mch, k_mcherry, C_mch = fitting.fit_decay_constant(time_axis, trace_result["mch_t_corr"])

        self.active_run_traces_cache[self.dataset_name] = {
            "time_axis": time_axis,
            "gfp_data": trace_result["gfp_t_corr"], "gfp_fit": (A_gfp, k_gfp, C_gfp),
            "has_gfp_control": trace_result["has_gfp_control"], "gfp_control": trace_result["gfp_control_norm"],
            "mch_data": trace_result["mch_t_corr"], "mch_fit": (A_mch, k_mcherry, C_mch),
            "has_mch_control": trace_result["has_mch_control"], "mch_control": trace_result["mch_control_norm"],
            "borrowed_control": trace_result["borrowed_control"],
        }

        self.save_cache_to_disk()

        new_rows = [
            {"Dataset": self.dataset_name, "Wavelength": self.wavelength, "Condition": self.condition, "Power_mW": self.power_mw, "Fluorophore": "GFP-LaminA", "Type": "Treated", "Decay_Constant_k": k_gfp},
            {"Dataset": self.dataset_name, "Wavelength": self.wavelength, "Condition": self.condition, "Power_mW": self.power_mw, "Fluorophore": "mCherry", "Type": "Treated", "Decay_Constant_k": k_mcherry},
        ]
        try:
            persistence.append_summary_rows(self.master_folder_path, self.dataset_name, new_rows)
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
            self.lbl_queue_progress.configure(text=f"Batch Complete ({len(self.dataset_prefixes)} sets) \U0001f389")
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

        treated_px = intensity_traces.total_mask_pixels(self.masks["TREATED"])
        control_px = intensity_traces.total_mask_pixels(self.masks["CONTROL"])
        bg_px = intensity_traces.total_mask_pixels(self.masks["BACKGROUND"])
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

    # ==============================================================================
    # EXPORT
    # ==============================================================================
    def export_publication_plots(self):
        df = persistence.read_summary_csv(self.master_folder_path)
        grouped_df, _ = pooling.pool_replicates(df)
        if grouped_df is None or grouped_df.empty: return

        target_dir = os.path.abspath(self.master_folder_path)
        individual_plots_dir = os.path.join(target_dir, "Individual_Decay_Plots")
        os.makedirs(individual_plots_dir, exist_ok=True)

        for name, traces in self.active_run_traces_cache.items():
            fig_ind = plotting.build_individual_decay_figure(name, traces)
            fig_ind.savefig(os.path.join(individual_plots_dir, f"{name}_decay_fit.png"), dpi=300)
            plt.close(fig_ind)

        fig1 = plotting.build_gfp_spectral_figure(grouped_df, dark_mode=False)
        fig1.savefig(os.path.join(target_dir, "pooled_gfp_laminA_spectral_decay.png"), dpi=300)
        plt.close(fig1)

        fig2 = plotting.build_mcherry_spectral_figure(grouped_df, dark_mode=False)
        fig2.savefig(os.path.join(target_dir, "pooled_mcherry_spectral_decay.png"), dpi=300)
        plt.close(fig2)

        fig3 = plotting.build_combined_spectral_figure(grouped_df, dark_mode=False)
        fig3.savefig(os.path.join(target_dir, "pooled_combined_spectral_kinetics.png"), dpi=300)
        plt.close(fig3)

        power_plots_dir = os.path.join(target_dir, "Power_Response_Plots")
        wavelengths_with_power = grouped_df[grouped_df['Power_Num'].notna()]['Wavelength_Num'].dropna().unique()
        if len(wavelengths_with_power) > 0:
            os.makedirs(power_plots_dir, exist_ok=True)
            for wl in sorted(wavelengths_with_power):
                fig_p = plotting.build_power_response_figure(grouped_df, wl, dark_mode=False)
                if fig_p is not None:
                    fig_p.savefig(os.path.join(power_plots_dir, f"power_response_{wl:g}nm.png"), dpi=300)
                    plt.close(fig_p)

        gc.collect()
        tk.messagebox.showinfo("Export Complete", f"Exported individual decay fits, global response matrices, and per-wavelength power-response plots to:\n{target_dir}")

    def on_closing(self):
        self.box_sync.stop()
        self.save_cache_to_disk()
        # Safe here because the entire application is shutting down.
        plt.close('all')
        gc.collect()
        self.quit()
        self.destroy()
