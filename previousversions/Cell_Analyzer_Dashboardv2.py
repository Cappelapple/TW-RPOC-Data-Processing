import os
import re
import numpy as np
import pandas as pd
import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu  # Added for automatic background exclusion

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==============================================================================
# MATHEMATICAL EXPONENTIAL FIT ENGINE
# ==============================================================================
def exponential_decay(t, A, k, C):
    return A * np.exp(-k * t) + C

def fit_decay_constant(time_axis, intensity_profile):
    try:
        amplitude_guess = intensity_profile[0] - np.min(intensity_profile)
        plateau_guess = np.min(intensity_profile)
        k_guess = 0.01 if abs(amplitude_guess) < 0.05 else 0.05
        
        p0 = [amplitude_guess, k_guess, plateau_guess]
        bounds = ([-2.0, 0.0, -0.5], [3.0, 20.0, 2.0])
        
        popt, _ = curve_fit(exponential_decay, time_axis, intensity_profile, p0=p0, bounds=bounds, maxfev=10000)
        # Return ALL THREE optimized pieces: A, k, C
        return popt[0], popt[1], popt[2]
    except:
        return np.nan, np.nan, np.nan
# ==============================================================================
# MAIN BATCH ANALYZER
# ==============================================================================
class AdvancedBatchCellAnalyzer(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Zhang Lab | Dynamic Local Thresholding Kinetics Dashboard")
        self.geometry("1500x900")
        
        self.dataset_prefixes = []
        self.current_dataset_index = -1
        self.master_folder_path = ""
        
        self.gfp_stack = None
        self.mcherry_stack = None
        self.mask_frame = None
        self.dataset_name = "None"
        self.wavelength = "Unknown"
        self.condition = "Unknown"
        self.x_pixels = 400
        self.y_pixels = 400
        self.num_frames = 40
        
        self.masks = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.patches = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}

        self.setup_ui_layout()

    def setup_ui_layout(self):
        self.grid_columnconfigure(0, weight=1)  
        self.grid_columnconfigure(1, weight=4)  
        self.grid_columnconfigure(2, weight=3)  
        self.grid_rowconfigure(0, weight=1)

        # PANEL 1: SIDEBAR CONTROLS
        self.sidebar = ctk.CTkFrame(self, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        lbl_title = ctk.CTkLabel(self.sidebar, text="Advanced Controls", font=ctk.CTkFont(size=16, weight="bold"))
        lbl_title.pack(pady=(15, 15), padx=20)

        self.btn_load_master = ctk.CTkButton(self.sidebar, text="Select Master Data Folder", command=self.load_flat_data_folder, fg_color="#1F6AA5")
        self.btn_load_master.pack(pady=5, padx=20, fill="x")

        self.lbl_queue_progress = ctk.CTkLabel(self.sidebar, text="Queue: 0/0 Completed", font=ctk.CTkFont(size=12, slant="italic"))
        self.lbl_queue_progress.pack(pady=(5, 2), padx=20, anchor="w")
        self.progress_bar = ctk.CTkProgressBar(self.sidebar)
        self.progress_bar.pack(pady=(0, 10), padx=20, fill="x")
        self.progress_bar.set(0)

        self.metadata_box = ctk.CTkTextbox(self.sidebar, height=110, activate_scrollbars=False, font=ctk.CTkFont(family="monospace", size=11))
        self.metadata_box.pack(pady=5, padx=20, fill="x")
        self.metadata_box.insert("0.0", "Queue Status: Idle\nDataset: None\nWave: --\nCond: --")
        self.metadata_box.configure(state="disabled")

        lbl_phase = ctk.CTkLabel(self.sidebar, text="Active Tracing Target:", font=ctk.CTkFont(weight="bold"))
        lbl_phase.pack(pady=(15, 2), padx=20, anchor="w")
        
        self.phase_var = tk.StringVar(value="TREATED")
        self.rdo_treated = ctk.CTkRadioButton(self.sidebar, text="Treated Cells (Red)", variable=self.phase_var, value="TREATED")
        self.rdo_treated.pack(pady=4, padx=30, anchor="w")
        self.rdo_control = ctk.CTkRadioButton(self.sidebar, text="Control Cells (Blue)", variable=self.phase_var, value="CONTROL")
        self.rdo_control.pack(pady=4, padx=30, anchor="w")
        self.rdo_bg = ctk.CTkRadioButton(self.sidebar, text="Background (Cyan)", variable=self.phase_var, value="BACKGROUND")
        self.rdo_bg.pack(pady=4, padx=30, anchor="w")

        self.btn_clear_current = ctk.CTkButton(self.sidebar, text="Clear Current Traces", fg_color="#7F8C8D", hover_color="#95A5A6", command=self.clear_current_traces)
        self.btn_clear_current.pack(pady=(20, 5), padx=20, fill="x")

        self.btn_undo = ctk.CTkButton(self.sidebar, text="Undo Last Shape", fg_color="#D9534F", hover_color="#C9302C", command=self.undo_last_shape)
        self.btn_undo.pack(pady=5, padx=20, fill="x")

        self.btn_prev = ctk.CTkButton(self.sidebar, text="← Previous Dataset", fg_color="#E67E22", hover_color="#D35400", command=self.navigate_to_previous_set)
        self.btn_prev.pack(pady=5, padx=20, fill="x")

        self.btn_process = ctk.CTkButton(self.sidebar, text="Save & Next Dataset →", font=ctk.CTkFont(weight="bold"), fg_color="#5CB85C", hover_color="#4CAE4C", command=self.process_and_advance_queue)
        self.btn_process.pack(pady=5, padx=20, fill="x")
        
        self.btn_export_plots = ctk.CTkButton(self.sidebar, text="💾 Export High-Res Charts", font=ctk.CTkFont(weight="bold"), fg_color="#8E44AD", hover_color="#7D3C98", command=self.export_publication_plots)
        self.btn_export_plots.pack(pady=(15, 5), padx=20, fill="x")

        # PANEL 2: CENTER WORKSPACE CANVAS
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.lbl_canvas_status = ctk.CTkLabel(self.canvas_frame, text="Load data folder to begin tracking.", font=ctk.CTkFont(slant="italic"))
        self.lbl_canvas_status.pack(expand=True)

        # PANEL 3: RIGHT TABBED RESULTS VIEWERS
        self.tabs_frame = ctk.CTkTabview(self)
        self.tabs_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        self.tab_current = self.tabs_frame.add("Current Fit Profiles")
        self.tab_global = self.tabs_frame.add("Global Comparative Summary")

    def load_flat_data_folder(self):
        self.master_folder_path = tk.filedialog.askdirectory(title="Select Folder Containing All Text Files")
        if not self.master_folder_path: return
        
        all_files = os.listdir(self.master_folder_path)
        prefixes = set()
        for f in all_files:
            if f.endswith(".txt"):
                base_prefix = re.split(r'_(GFP|mCherry|Mask|Parameter|Parameters)', f, flags=re.IGNORECASE)[0]
                prefixes.add(base_prefix)
                
        valid_prefixes = []
        for p in prefixes:
            has_gfp = any(f.lower() == f"{p.lower()}_gfp.txt" for f in all_files)
            has_mcherry = any(f.lower() == f"{p.lower()}_mcherry.txt" for f in all_files)
            has_mask = any(f.lower() == f"{p.lower()}_mask.txt" for f in all_files)
            if has_gfp and has_mcherry and has_mask:
                valid_prefixes.append(p)
                
        if not valid_prefixes:
            tk.messagebox.showerror("Error", "No valid matching structural combinations found.")
            return
            
        self.dataset_prefixes = sorted(valid_prefixes)
        self.current_dataset_index = 0
        self.initialize_dataset_at_index(self.current_dataset_index)
        self.render_global_comparison_chart()

    def initialize_dataset_at_index(self, index):
        if index < 0 or index >= len(self.dataset_prefixes): return
        prefix = self.dataset_prefixes[index]
        self.dataset_name = prefix
        
        all_files = os.listdir(self.master_folder_path)
        gfp_file = next(f for f in all_files if f.lower() == f"{prefix.lower()}_gfp.txt")
        mcherry_file = next(f for f in all_files if f.lower() == f"{prefix.lower()}_mcherry.txt")
        mask_file = next(f for f in all_files if f.lower() == f"{prefix.lower()}_mask.txt")
        param_file = next((f for f in all_files if f.lower() == f"{prefix.lower()}_parameters.txt" or f.lower() == f"{prefix.lower()}_parameter.txt"), None)

        if param_file:
            with open(os.path.join(self.master_folder_path, param_file), 'r') as f:
                content = f.read()
            x_match = re.search(r'# of Pixels:\s*(\d+)', content)
            frame_match = re.search(r'# of Images:\s*(\d+)', content)
            self.x_pixels = int(x_match.group(1)) if x_match else 400
            self.y_pixels = self.x_pixels
            self.num_frames = int(frame_match.group(1)) if frame_match else 40

        wave_match = re.search(r'(\d+nm)', prefix, re.IGNORECASE)
        self.wavelength = wave_match.group(1) if wave_match else "Unknown"
        self.condition = "Normoxia" if "norm" in prefix.lower() else "Hypoxia" if "hypo" in prefix.lower() else "Unknown"

        def parse_data_file(name):
            with open(os.path.join(self.master_folder_path, name), 'r') as f:
                content = f.read()
            return np.array(re.findall(r'[-+]?\d*\.\d+|\d+', content), dtype=np.float64)

        gfp_data = parse_data_file(gfp_file)
        mcherry_data = parse_data_file(mcherry_file)
        mask_data = parse_data_file(mask_file)

        pts_per_frame = self.x_pixels * self.y_pixels
        self.gfp_stack = gfp_data[:self.num_frames * pts_per_frame].reshape((self.num_frames, self.y_pixels, self.x_pixels))
        self.mcherry_stack = mcherry_data[:self.num_frames * pts_per_frame].reshape((self.num_frames, self.y_pixels, self.x_pixels))
        self.mask_frame = mask_data[:pts_per_frame].reshape((self.y_pixels, self.x_pixels))

        total_sets = len(self.dataset_prefixes)
        self.lbl_queue_progress.configure(text=f"Batch Progress: {index + 1} / {total_sets} Sets")
        self.progress_bar.set((index) / total_sets)

        self.metadata_box.configure(state="normal")
        self.metadata_box.delete("0.0", "end")
        self.metadata_box.insert("0.0", f"Active Run [{index+1}/{total_sets}]\nPrefix: {self.dataset_name}\nWave: {self.wavelength}\nCond: {self.condition}\nDim: {self.x_pixels}x{self.y_pixels}")
        self.metadata_box.configure(state="disabled")

        self.clear_current_traces()

    def clear_current_traces(self):
        self.masks = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.patches = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.phase_var.set("TREATED")
        self.build_interactive_canvas()

    def navigate_to_previous_set(self):
        if self.current_dataset_index > 0:
            self.current_dataset_index -= 1
            self.initialize_dataset_at_index(self.current_dataset_index)
        else:
            tk.messagebox.showinfo("Queue Info", "Already at the beginning of the workspace queue.")

    def build_interactive_canvas(self):
        for widget in self.canvas_frame.winfo_children(): widget.destroy()
        self.fig_canvas, self.ax_canvas = plt.subplots(figsize=(6, 6), facecolor="#2B2B2B")
        self.ax_canvas.imshow(self.gfp_stack[0], cmap='gray')
        display_mask = np.where(self.mask_frame > (np.max(self.mask_frame) * 0.1), 1, 0)
        self.ax_canvas.imshow(display_mask, cmap='Reds', alpha=0.15)
        self.ax_canvas.axis('off')
        self.fig_canvas.tight_layout()
        self.tk_canvas = FigureCanvasTkAgg(self.fig_canvas, master=self.canvas_frame)
        self.tk_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.lasso = LassoSelector(self.ax_canvas, self.on_lasso_release, props=dict(color='yellow', linewidth=1.5))
        self.tk_canvas.draw()

    # ==============================================================================
    # AUTOMATIC SEGMENATION AND THRESHOLDING INTERSECTION
    # ==============================================================================
    def on_lasso_release(self, verts):
        if len(verts) < 3: return
        phase = self.phase_var.get()
        color = 'r' if phase == "TREATED" else 'b' if phase == "CONTROL" else 'c'
        
        y_grid, x_grid = np.mgrid[:self.y_pixels, :self.x_pixels]
        points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
        
        # 1. Map raw hand-drawn boundary mask
        raw_hand_mask = Path(verts).contains_points(points).reshape((self.y_pixels, self.x_pixels))
        final_processed_mask = raw_hand_mask
        
        # 2. Apply Otsu Threshold Filtering (Only for Treated/Control cells)
        if phase in ["TREATED", "CONTROL"]:
            local_pixels = self.mcherry_stack[0][raw_hand_mask]
            
            if local_pixels.size > 5:
                try:
                    thresh_val = threshold_otsu(local_pixels)
                    intensity_mask = self.mcherry_stack[0] >= thresh_val
                    candidate_mask = np.logical_and(raw_hand_mask, intensity_mask)
                    
                    # SAFETY CHECK: Only use thresholded mask if it didn't wipe out the whole cell
                    if np.sum(candidate_mask) > 0:
                        final_processed_mask = candidate_mask
                    else:
                        final_processed_mask = raw_hand_mask # Fallback to full hand trace
                except:
                    final_processed_mask = raw_hand_mask # Fallback on error
                    
        # Plot remaining pixels or boundaries on screen
        y_idx, x_idx = np.where(final_processed_mask)
        poly = Polygon(verts, edgecolor=color, facecolor='none', linewidth=1.2, linestyle=':')
        self.ax_canvas.add_patch(poly)
        
        if y_idx.size > 0:
            self.ax_canvas.scatter(x_idx, y_idx, color=color, s=1, alpha=0.4)
        
        self.masks[phase].append(final_processed_mask)
        self.patches[phase].append(poly)
        self.tk_canvas.draw_idle()

    def undo_last_shape(self):
        phase = self.phase_var.get()
        if self.masks[phase]:
            self.masks[phase].pop()
            self.patches[phase].pop().remove()
            self.build_interactive_canvas() # Complete re-draw to clear scatter dots cleanly

    def process_and_advance_queue(self):
        if not self.masks["BACKGROUND"] or not self.masks["TREATED"]:
            tk.messagebox.showwarning("Warning", "Incomplete traces! Select Treated and Background areas.")
            return

        def get_mask_average_trace(stack, masks_list):
            intensities = []
            for frame in stack:
                frame_vals = [np.mean(frame[mask]) for mask in masks_list if np.any(mask)]
                intensities.append(np.mean(frame_vals) if frame_vals else 1.0)
            return np.array(intensities)

        gfp_t = get_mask_average_trace(self.gfp_stack, self.masks["TREATED"])
        gfp_c = get_mask_average_trace(self.gfp_stack, self.masks["CONTROL"]) if self.masks["CONTROL"] else np.ones(self.num_frames)
        gfp_bg = get_mask_average_trace(self.gfp_stack, self.masks["BACKGROUND"])
        mcherry_t = get_mask_average_trace(self.mcherry_stack, self.masks["TREATED"])
        mcherry_c = get_mask_average_trace(self.mcherry_stack, self.masks["CONTROL"]) if self.masks["CONTROL"] else np.ones(self.num_frames)
        mcherry_bg = get_mask_average_trace(self.mcherry_stack, self.masks["BACKGROUND"])

        gfp_t_corr = (gfp_t - gfp_bg) / (gfp_t[0] - gfp_bg[0])
        gfp_c_corr = (gfp_c - gfp_bg) / (gfp_c[0] - gfp_bg[0]) if self.masks["CONTROL"] else np.ones(self.num_frames)
        mcherry_t_corr = (mcherry_t - mcherry_bg) / (mcherry_t[0] - mcherry_bg[0])
        mcherry_c_corr = (mcherry_c - mcherry_bg) / (mcherry_c[0] - mcherry_bg[0]) if self.masks["CONTROL"] else np.ones(self.num_frames)

        time_axis = np.arange(self.num_frames)
        # Unpack all three optimized values
        A_gfp, k_gfp, C_gfp = fit_decay_constant(time_axis, gfp_t_corr)
        A_mch, k_mcherry, C_mch = fit_decay_constant(time_axis, mcherry_t_corr)

        # Log to spreadsheet (we still log just the rate constant 'k')
        summary_file = os.path.join(self.master_folder_path, "master_kinetics_summary.csv")
        new_rows = [
            {"Dataset": self.dataset_name, "Wavelength": self.wavelength, "Condition": self.condition, "Fluorophore": "GFP", "Type": "Treated", "Decay_Constant_k": k_gfp},
            {"Dataset": self.dataset_name, "Wavelength": self.wavelength, "Condition": self.condition, "Fluorophore": "mCherry", "Type": "Treated", "Decay_Constant_k": k_mcherry}
        ]
        # ... [keep your existing spreadsheet saving code here] ...

        # --- Tab 1 fit plots drawing update ---
        for widget in self.tab_current.winfo_children(): widget.destroy()
        fig_fit, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), facecolor="#2B2B2B")
        ax1.set_facecolor("#1E1E1E"); ax2.set_facecolor("#1E1E1E")

        # Plot GFP using its true mathematical parameters
        ax1.plot(time_axis, gfp_t_corr, 'go', alpha=0.4, label="Data")
        if not np.isnan(k_gfp): 
            # USE TRUE A, k, AND C FOUND BY PYTHON
            ax1.plot(time_axis, exponential_decay(time_axis, A_gfp, k_gfp, C_gfp), 'g-', label=f"k={k_gfp:.4f}")
        if self.masks["CONTROL"]: ax1.plot(time_axis, gfp_c_corr, 'g--', label="Control")
        ax1.set_title("GFP Profiles", color="white", fontsize=10); ax1.tick_params(colors="white"); ax1.legend(labelcolor="white", facecolor="#2B2B2B", edgecolor="none")

        # Plot mCherry using its true mathematical parameters
        ax2.plot(time_axis, mcherry_t_corr, 'ro', alpha=0.4, label="Data")
        if not np.isnan(k_mcherry): 
            # USE TRUE A, k, AND C FOUND BY PYTHON
            ax2.plot(time_axis, exponential_decay(time_axis, A_mch, k_mcherry, C_mch), 'r-', label=f"k={k_mcherry:.4f}")
        if self.masks["CONTROL"]: ax2.plot(time_axis, mcherry_c_corr, 'r--', label="Control")
        ax2.set_title("mCherry Profiles", color="white", fontsize=10); ax2.tick_params(colors="white"); ax2.legend(labelcolor="white", facecolor="#2B2B2B", edgecolor="none")

        fig_fit.tight_layout()
        canvas_fit = FigureCanvasTkAgg(fig_fit, master=self.tab_current)
        canvas_fit.get_tk_widget().pack(fill="both", expand=True)
        canvas_fit.draw()

        self.render_global_comparison_chart()
        self.tabs_frame.set("Current Fit Profiles")
        self.update_idletasks()

        self.current_dataset_index += 1
        if self.current_dataset_index < len(self.dataset_prefixes):
            self.initialize_dataset_at_index(self.current_dataset_index)
        else:
            self.progress_bar.set(1.0)
            self.lbl_queue_progress.configure(text="Batch Run Complete! 🎉")
            tk.messagebox.showinfo("Done", "All file sets successfully cataloged.")

    def generate_plots_engine(self, dark_mode=True):
        summary_file = os.path.join(self.master_folder_path, "master_kinetics_summary.csv")
        if not os.path.exists(summary_file) or os.path.getsize(summary_file) < 10: return None
        
        df = pd.read_csv(summary_file)
        df['Wavelength_Num'] = df['Wavelength'].astype(str).str.extract(r'(\d+)').astype(float)
        df = df.dropna(subset=['Wavelength_Num', 'Decay_Constant_k'])
        
        gfp_df = df[df["Fluorophore"] == "GFP"]
        mch_df = df[df["Fluorophore"] == "mCherry"]
        marker_dict = {"Normoxia": "o", "Hypoxia": "s", "Unknown": "^"}
        
        bg_col = "#2B2B2B" if dark_mode else "white"
        face_col = "#1E1E1E" if dark_mode else "#F5F5F5"
        txt_col = "white" if dark_mode else "black"
        grid_col = "gray" if dark_mode else "darkgray"

        fig_global = plt.figure(figsize=(11, 8), facecolor=bg_col)
        
        # CHART 1: GFP Only (Normoxia vs Hypoxia 2 Lines)
        ax1 = plt.subplot2grid((2, 2), (0, 0))
        ax1.set_facecolor(face_col)
        if not gfp_df.empty:
            for cond, grp in gfp_df.groupby("Condition"):
                ax1.scatter(grp['Wavelength_Num'], grp['Decay_Constant_k'], color='#5CB85C' if cond=="Normoxia" else '#2E7D32', marker=marker_dict.get(cond, "o"), s=80, alpha=0.8, edgecolors=txt_col, label=f"GFP {cond}")
            for cond, color_fit in [("Normoxia", "#5CB85C"), ("Hypoxia", "#2E7D32")]:
                sub_df = gfp_df[gfp_df["Condition"] == cond].sort_values('Wavelength_Num')
                if len(sub_df) >= 2:
                    p = np.polyfit(sub_df['Wavelength_Num'], sub_df['Decay_Constant_k'], 1)
                    x_axis = np.linspace(df['Wavelength_Num'].min() - 15, df['Wavelength_Num'].max() + 15, 100)
                    ax1.plot(x_axis, np.polyval(p, x_axis), color=color_fit, linewidth=2, label=f"{cond} Fit")
        ax1.set_title("GFP Spectral Kinetics", color=txt_col, fontsize=11)
        ax1.set_ylabel("Decay Value (k)", color=txt_col); ax1.tick_params(colors=txt_col); ax1.grid(True, linestyle=":", color=grid_col, alpha=0.3)
        l1 = ax1.legend(facecolor=bg_col, edgecolor="none", loc="best")
        if l1: [t.set_color(txt_col) for t in l1.get_texts()]

        # CHART 2: mCherry Only (Normoxia vs Hypoxia 2 Lines)
        ax2 = plt.subplot2grid((2, 2), (0, 1))
        ax2.set_facecolor(face_col)
        if not mch_df.empty:
            for cond, grp in mch_df.groupby("Condition"):
                ax2.scatter(grp['Wavelength_Num'], grp['Decay_Constant_k'], color='#D9534F' if cond=="Normoxia" else '#C0392B', marker=marker_dict.get(cond, "o"), s=80, alpha=0.8, edgecolors=txt_col, label=f"mCherry {cond}")
            for cond, color_fit in [("Normoxia", "#D9534F"), ("Hypoxia", "#C0392B")]:
                sub_df = mch_df[mch_df["Condition"] == cond].sort_values('Wavelength_Num')
                if len(sub_df) >= 2:
                    p = np.polyfit(sub_df['Wavelength_Num'], sub_df['Decay_Constant_k'], 1)
                    x_axis = np.linspace(df['Wavelength_Num'].min() - 15, df['Wavelength_Num'].max() + 15, 100)
                    ax2.plot(x_axis, np.polyval(p, x_axis), color=color_fit, linewidth=2, label=f"{cond} Fit")
        ax2.set_title("mCherry Spectral Kinetics", color=txt_col, fontsize=11)
        ax2.tick_params(colors=txt_col); ax2.grid(True, linestyle=":", color=grid_col, alpha=0.3)
        l2 = ax2.legend(facecolor=bg_col, edgecolor="none", loc="best")
        if l2: [t.set_color(txt_col) for t in l2.get_texts()]

        # CHART 3: COMBINED VIEW (All 4 Trends Together)
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
        ax3.set_facecolor(face_col)
        color_map = {
            ("GFP", "Normoxia"): "#5CB85C",   ("GFP", "Hypoxia"): "#2E7D32",
            ("mCherry", "Normoxia"): "#D9534F", ("mCherry", "Hypoxia"): "#C0392B"
        }
        for (fl, cond), grp in df.groupby(["Fluorophore", "Condition"]):
            c = color_map.get((fl, cond), "blue")
            ax3.scatter(grp['Wavelength_Num'], grp['Decay_Constant_k'], color=c, marker=marker_dict.get(cond, "o"), s=85, alpha=0.8, edgecolors=txt_col)
            if len(grp) >= 2:
                grp = grp.sort_values(by='Wavelength_Num')
                p = np.polyfit(grp['Wavelength_Num'], grp['Decay_Constant_k'], 1)
                x_axis = np.linspace(df['Wavelength_Num'].min() - 20, df['Wavelength_Num'].max() + 20, 100)
                ax3.plot(x_axis, np.polyval(p, x_axis), color=c, linewidth=2.5, label=f"{fl} [{cond}] Fit")
                
        ax3.set_title("Unified Cross-Matrix Footprint Summary (All 4 Series)", color=txt_col, fontsize=11)
        ax3.set_xlabel("Wavelength Track (nm)", color=txt_col); ax3.set_ylabel("Rate Constant (k)", color=txt_col)
        ax3.tick_params(colors=txt_col); ax3.grid(True, linestyle=":", color=grid_col, alpha=0.3)
        l3 = ax3.legend(facecolor=bg_col, edgecolor="none", loc="best")
        if l3: [t.set_color(txt_col) for t in l3.get_texts()]

        fig_global.tight_layout()
        return fig_global

    def render_global_comparison_chart(self):
        for widget in self.tab_global.winfo_children(): widget.destroy()
        fig = self.generate_plots_engine(dark_mode=True)
        if fig is None: return
        canvas_global = FigureCanvasTkAgg(fig, master=self.tab_global)
        canvas_global.get_tk_widget().pack(fill="both", expand=True)
        canvas_global.draw()

    def export_publication_plots(self):
        summary_file = os.path.join(self.master_folder_path, "master_kinetics_summary.csv")
        if not os.path.exists(summary_file): return
        
        df = pd.read_csv(summary_file)
        df['Wavelength_Num'] = df['Wavelength'].astype(str).str.extract(r'(\d+)').astype(float)
        df = df.dropna(subset=['Wavelength_Num', 'Decay_Constant_k'])
        
        gfp_df = df[df["Fluorophore"] == "GFP"]
        mch_df = df[df["Fluorophore"] == "mCherry"]
        marker_dict = {"Normoxia": "o", "Hypoxia": "s", "Unknown": "^"}
        
        # 1. GFP white background save
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        if not gfp_df.empty:
            for cond, grp in gfp_df.groupby("Condition"):
                ax1.scatter(grp['Wavelength_Num'], grp['Decay_Constant_k'], color='#5CB85C' if cond=="Normoxia" else '#2E7D32', marker=marker_dict.get(cond, "o"), s=100, label=f"GFP {cond}")
            for cond, col in [("Normoxia", "#5CB85C"), ("Hypoxia", "#2E7D32")]:
                sub = gfp_df[gfp_df["Condition"] == cond].sort_values('Wavelength_Num')
                if len(sub) >= 2:
                    p = np.polyfit(sub['Wavelength_Num'], sub['Decay_Constant_k'], 1)
                    x = np.linspace(df['Wavelength_Num'].min()-15, df['Wavelength_Num'].max()+15, 100)
                    ax1.plot(x, np.polyval(p, x), color=col, linewidth=2, label=f"{cond} Fit")
        ax1.set_title("GFP Photobleaching Response"); ax1.set_xlabel("Wavelength (nm)"); ax1.set_ylabel("k (s^-1)"); ax1.grid(True); ax1.legend()
        fig1.tight_layout(); fig1.savefig(os.path.join(self.master_folder_path, "gfp_spectral_decay.png"), dpi=300); plt.close(fig1)

        # 2. mCherry white background save
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        if not mch_df.empty:
            for cond, grp in mch_df.groupby("Condition"):
                ax2.scatter(grp['Wavelength_Num'], grp['Decay_Constant_k'], color='#D9534F' if cond=="Normoxia" else '#C0392B', marker=marker_dict.get(cond, "o"), s=100, label=f"mCherry {cond}")
            for cond, col in [("Normoxia", "#D9534F"), ("Hypoxia", "#C0392B")]:
                sub = mch_df[mch_df["Condition"] == cond].sort_values('Wavelength_Num')
                if len(sub) >= 2:
                    p = np.polyfit(sub['Wavelength_Num'], sub['Decay_Constant_k'], 1)
                    x = np.linspace(df['Wavelength_Num'].min()-15, df['Wavelength_Num'].max()+15, 100)
                    ax2.plot(x, np.polyval(p, x), color=col, linewidth=2, label=f"{cond} Fit")
        ax2.set_title("mCherry Photobleaching Response"); ax2.set_xlabel("Wavelength (nm)"); ax2.set_ylabel("k (s^-1)"); ax2.grid(True); ax2.legend()
        fig2.tight_layout(); fig2.savefig(os.path.join(self.master_folder_path, "mcherry_spectral_decay.png"), dpi=300); plt.close(fig2)

        # 3. Combined white background save (4 Separate Regression Fit Lines)
        fig3, ax3 = plt.subplots(figsize=(8, 5.5))
        color_map = {
            ("GFP", "Normoxia"): "#5CB85C",   ("GFP", "Hypoxia"): "#2E7D32",
            ("mCherry", "Normoxia"): "#D9534F", ("mCherry", "Hypoxia"): "#C0392B"
        }
        for (fl, cond), grp in df.groupby(["Fluorophore", "Condition"]):
            c = color_map.get((fl, cond), "blue")
            ax3.scatter(grp['Wavelength_Num'], grp['Decay_Constant_k'], color=c, marker=marker_dict.get(cond, "o"), s=100)
            if len(grp) >= 2:
                grp = grp.sort_values(by='Wavelength_Num')
                p = np.polyfit(grp['Wavelength_Num'], grp['Decay_Constant_k'], 1)
                x = np.linspace(df['Wavelength_Num'].min()-20, df['Wavelength_Num'].max()+20, 100)
                ax3.plot(x, np.polyval(p, x), color=c, linewidth=2, label=f"{fl} [{cond}] Fit Line")
        ax3.set_title("Unified Spectral Response Matrix (4 Independent Series)")
        ax3.set_xlabel("Wavelength (nm)"); ax3.set_ylabel("Decay Rate Constant k (s^-1)"); ax3.grid(True); ax3.legend()
        fig3.tight_layout(); fig3.savefig(os.path.join(self.master_folder_path, "combined_spectral_kinetics.png"), dpi=300); plt.close(fig3)

        tk.messagebox.showinfo("Export Complete", "Saved 3 clean publication figures inside your workspace folder!")

if __name__ == "__main__":
    app = AdvancedBatchCellAnalyzer()
    app.mainloop()