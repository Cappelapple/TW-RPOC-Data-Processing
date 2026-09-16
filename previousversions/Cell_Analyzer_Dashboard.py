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

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ==============================================================================
# MATHEMATICAL EXPONENTIAL FIT ENGINE
# ==============================================================================
def exponential_decay(t, A, k, C):
    return A * np.exp(-k * t) + C

def fit_decay_constant(time_axis, intensity_profile):
    try:
        p0 = [intensity_profile[0] - intensity_profile[-1], 0.1, intensity_profile[-1]]
        bounds = ([0, 0, 0], [2.0, 5.0, 1.5])
        popt, _ = curve_fit(exponential_decay, time_axis, intensity_profile, p0=p0, bounds=bounds, maxfev=5000)
        return popt[1]
    except:
        return np.nan

# ==============================================================================
# BATCH WORKFLOW USER INTERFACE (FLAT FOLDER MODE)
# ==============================================================================
class FlatFolderBatchAnalyzer(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Zhang Lab | Flat-Folder Batch Kinetics Dashboard")
        self.geometry("1450x870")
        
        # Batch processing structures
        self.dataset_prefixes = []
        self.current_dataset_index = -1
        self.master_folder_path = ""
        
        # Core active data variables
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

        # PANEL 1: SIDEBAR
        self.sidebar = ctk.CTkFrame(self, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        lbl_title = ctk.CTkLabel(self.sidebar, text="Flat-Folder Controls", font=ctk.CTkFont(size=16, weight="bold"))
        lbl_title.pack(pady=(20, 20), padx=20)

        self.btn_load_master = ctk.CTkButton(self.sidebar, text="Select Master Data Folder", command=self.load_flat_data_folder, fg_color="#1F6AA5")
        self.btn_load_master.pack(pady=10, padx=20, fill="x")

        self.lbl_queue_progress = ctk.CTkLabel(self.sidebar, text="Queue: 0/0 Completed", font=ctk.CTkFont(size=12, slant="italic"))
        self.lbl_queue_progress.pack(pady=(5, 5), padx=20, anchor="w")
        self.progress_bar = ctk.CTkProgressBar(self.sidebar)
        self.progress_bar.pack(pady=(0, 15), padx=20, fill="x")
        self.progress_bar.set(0)

        self.metadata_box = ctk.CTkTextbox(self.sidebar, height=120, activate_scrollbars=False, font=ctk.CTkFont(family="monospace", size=11))
        self.metadata_box.pack(pady=10, padx=20, fill="x")
        self.metadata_box.insert("0.0", "Queue Status: Idle\nDataset: None\nWave: --\nCond: --")
        self.metadata_box.configure(state="disabled")

        lbl_phase = ctk.CTkLabel(self.sidebar, text="Active Tracing Target:", font=ctk.CTkFont(weight="bold"))
        lbl_phase.pack(pady=(20, 5), padx=20, anchor="w")
        
        self.phase_var = tk.StringVar(value="TREATED")
        self.rdo_treated = ctk.CTkRadioButton(self.sidebar, text="Treated Cells (Red)", variable=self.phase_var, value="TREATED")
        self.rdo_treated.pack(pady=5, padx=30, anchor="w")
        self.rdo_control = ctk.CTkRadioButton(self.sidebar, text="Control Cells (Blue)", variable=self.phase_var, value="CONTROL")
        self.rdo_control.pack(pady=5, padx=30, anchor="w")
        self.rdo_bg = ctk.CTkRadioButton(self.sidebar, text="Background (Cyan)", variable=self.phase_var, value="BACKGROUND")
        self.rdo_bg.pack(pady=5, padx=30, anchor="w")

        self.btn_undo = ctk.CTkButton(self.sidebar, text="Undo Last Shape", fg_color="#D9534F", hover_color="#C9302C", command=self.undo_last_shape)
        self.btn_undo.pack(pady=(35, 10), padx=20, fill="x")

        self.btn_process = ctk.CTkButton(self.sidebar, text="Save & Next Dataset", fg_color="#5CB85C", hover_color="#4CAE4C", command=self.process_and_advance_queue)
        self.btn_process.pack(pady=10, padx=20, fill="x")

        # PANEL 2: CENTER CANVAS
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.lbl_canvas_status = ctk.CTkLabel(self.canvas_frame, text="Load your primary folder to auto-group datasets.", font=ctk.CTkFont(slant="italic"))
        self.lbl_canvas_status.pack(expand=True)

        # PANEL 3: RIGHT TABS
        self.tabs_frame = ctk.CTkTabview(self)
        self.tabs_frame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        self.tab_current = self.tabs_frame.add("Current Fit Profiles")
        self.tab_global = self.tabs_frame.add("Global Comparative Summary")
        
        self.lbl_tab1_status = ctk.CTkLabel(self.tab_current, text="Awaiting active traces...", font=ctk.CTkFont(slant="italic"))
        self.lbl_tab1_status.pack(expand=True)
        self.lbl_tab2_status = ctk.CTkLabel(self.tab_global, text="Awaiting summary data file...", font=ctk.CTkFont(slant="italic"))
        self.lbl_tab2_status.pack(expand=True)

    # ==============================================================================
    # FILENAME PREFIX PARSING ENGINE
    # ==============================================================================
    def load_flat_data_folder(self):
        self.master_folder_path = tk.filedialog.askdirectory(title="Select Folder Containing All Text Files")
        if not self.master_folder_path: return
        
        all_files = os.listdir(self.master_folder_path)
        
        # Isolate all base names by finding everything before "_GFP", "_mCherry", etc.
        prefixes = set()
        for f in all_files:
            if f.endswith(".txt"):
                # Split at known channel endings to capture the true prefix
                base_prefix = re.split(r'_(GFP|mCherry|Mask|Parameter)', f, flags=re.IGNORECASE)[0]
                prefixes.add(base_prefix)
                
        # Filter prefixes to ensure they actually have matching GFP, mCherry, and Mask files present
        valid_prefixes = []
        for p in prefixes:
            has_gfp = any(f.lower() == f"{p.lower()}_gfp.txt" for f in all_files)
            has_mcherry = any(f.lower() == f"{p.lower()}_mcherry.txt" for f in all_files)
            has_mask = any(f.lower() == f"{p.lower()}_mask.txt" for f in all_files)
            if has_gfp and has_mcherry and has_mask:
                valid_prefixes.append(p)
                
        if not valid_prefixes:
            tk.messagebox.showerror("Error", "No complete matching sets (GFP, mCherry, and Mask) discovered using filename conventions.")
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
        # Find exact case-matching files inside directory
        gfp_file = next(f for f in all_files if f.lower() == f"{prefix.lower()}_gfp.txt")
        mcherry_file = next(f for f in all_files if f.lower() == f"{prefix.lower()}_mcherry.txt")
        mask_file = next(f for f in all_files if f.lower() == f"{prefix.lower()}_mask.txt")
        param_file = next((f for f in all_files if f.lower() == f"{prefix.lower()}_parameters.txt" or f.lower() == f"{prefix.lower()}_parameter.txt"), None)

        # Parse parameter configurations if file is present
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

        self.masks = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.patches = {"TREATED": [], "CONTROL": [], "BACKGROUND": []}
        self.phase_var.set("TREATED")
        
        self.build_interactive_canvas()

    # ==============================================================================
    # CANVAS AND PROCESSING LIFECYCLES
    # ==============================================================================
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

    def on_lasso_release(self, verts):
        if len(verts) < 3: return
        phase = self.phase_var.get()
        color = 'r' if phase == "TREATED" else 'b' if phase == "CONTROL" else 'c'
        poly = Polygon(verts, edgecolor=color, facecolor='none', linewidth=1.8)
        self.ax_canvas.add_patch(poly)
        y_grid, x_grid = np.mgrid[:self.y_pixels, :self.x_pixels]
        points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
        grid_mask = Path(verts).contains_points(points).reshape((self.y_pixels, self.x_pixels))
        self.masks[phase].append(grid_mask)
        self.patches[phase].append(poly)
        self.tk_canvas.draw_idle()

    def undo_last_shape(self):
        phase = self.phase_var.get()
        if self.masks[phase]:
            self.masks[phase].pop()
            self.patches[phase].pop().remove()
            self.tk_canvas.draw_idle()

    def process_and_advance_queue(self):
        if not self.masks["BACKGROUND"] or not self.masks["TREATED"]:
            tk.messagebox.showwarning("Warning", "Incomplete traces! Add 'Treated' and 'Background' selections.")
            return

        def get_mask_average_trace(stack, masks_list):
            intensities = []
            for frame in stack:
                frame_vals = [np.mean(frame[mask]) for mask in masks_list]
                intensities.append(np.mean(frame_vals))
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
        k_gfp = fit_decay_constant(time_axis, gfp_t_corr)
        k_mcherry = fit_decay_constant(time_axis, mcherry_t_corr)

        summary_file = os.path.join(self.master_folder_path, "master_kinetics_summary.csv")
        new_rows = [
            {"Dataset": self.dataset_name, "Wavelength": self.wavelength, "Condition": self.condition, "Fluorophore": "GFP", "Type": "Treated", "Decay_Constant_k": k_gfp},
            {"Dataset": self.dataset_name, "Wavelength": self.wavelength, "Condition": self.condition, "Fluorophore": "mCherry", "Type": "Treated", "Decay_Constant_k": k_mcherry}
        ]
        summary_df = pd.DataFrame(new_rows)
        if os.path.exists(summary_file):
            master_df = pd.read_csv(summary_file)
            master_df = master_df[master_df["Dataset"] != self.dataset_name]
            master_df = pd.concat([master_df, summary_df], ignore_index=True)
        else:
            master_df = summary_df
        master_df.to_csv(summary_file, index=False)

        # Tab 1 Overlay plots
        for widget in self.tab_current.winfo_children(): widget.destroy()
        fig_fit, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), facecolor="#2B2B2B")
        ax1.set_facecolor("#1E1E1E"); ax2.set_facecolor("#1E1E1E")

        ax1.plot(time_axis, gfp_t_corr, 'go', alpha=0.4)
        if not np.isnan(k_gfp): ax1.plot(time_axis, exponential_decay(time_axis, gfp_t_corr[0]-gfp_t_corr[-1], k_gfp, gfp_t_corr[-1]), 'g-', label=f"k={k_gfp:.4f}")
        if self.masks["CONTROL"]: ax1.plot(time_axis, gfp_c_corr, 'g--')
        ax1.set_title("GFP Decays", color="white", fontsize=10); ax1.tick_params(colors="white"); ax1.legend()

        ax2.plot(time_axis, mcherry_t_corr, 'ro', alpha=0.4)
        if not np.isnan(k_mcherry): ax2.plot(time_axis, exponential_decay(time_axis, mcherry_t_corr[0]-mcherry_t_corr[-1], k_mcherry, mcherry_t_corr[-1]), 'r-', label=f"k={k_mcherry:.4f}")
        if self.masks["CONTROL"]: ax2.plot(time_axis, mcherry_c_corr, 'r--')
        ax2.set_title("mCherry Decays", color="white", fontsize=10); ax2.tick_params(colors="white"); ax2.legend()

        fig_fit.tight_layout()
        canvas_fit = FigureCanvasTkAgg(fig_fit, master=self.tab_current)
        canvas_fit.get_tk_widget().pack(fill="both", expand=True)
        canvas_fit.draw()

        self.render_global_comparison_chart()
        self.tabs_frame.set("Current Fit Profiles")

        # Advance Position index
        self.current_dataset_index += 1
        if self.current_dataset_index < len(self.dataset_prefixes):
            self.initialize_dataset_at_index(self.current_dataset_index)
        else:
            self.progress_bar.set(1.0)
            self.lbl_queue_progress.configure(text="Batch Run Complete! 🎉")
            tk.messagebox.showinfo("Done", "All file sets in this directory have been processed and saved.")

    def render_global_comparison_chart(self):
        summary_file = os.path.join(self.master_folder_path, "master_kinetics_summary.csv")
        if not os.path.exists(summary_file) or os.path.getsize(summary_file) < 10: return

        for widget in self.tab_global.winfo_children(): widget.destroy()
        df = pd.read_csv(summary_file)
        
        fig_global, ax = plt.subplots(figsize=(7, 4), facecolor="#2B2B2B")
        ax.set_facecolor("#1E1E1E")
        
        sns.barplot(data=df, x="Wavelength", y="Decay_Constant_k", hue="Condition", palette="muted", errorbar=None, ax=ax)
        ax.set_title("Global Photobleaching Summary Matrix", color="white", fontsize=11)
        ax.set_xlabel("Wavelength", color="white"); ax.set_ylabel("Decay Value (k)", color="white")
        ax.tick_params(colors="white"); ax.grid(True, linestyle=":", color="gray", alpha=0.3)
        legend = ax.legend(facecolor="#2B2B2B", edgecolor="none")
        for text in legend.get_texts(): text.set_color("white")

        fig_global.tight_layout()
        canvas_global = FigureCanvasTkAgg(fig_global, master=self.tab_global)
        canvas_global.get_tk_widget().pack(fill="both", expand=True)
        canvas_global.draw()

if __name__ == "__main__":
    app = FlatFolderBatchAnalyzer()
    app.mainloop()