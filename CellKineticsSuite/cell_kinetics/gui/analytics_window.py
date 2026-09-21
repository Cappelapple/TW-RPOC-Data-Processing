"""Pop-up analytics dashboard: per-dataset fit preview, pooled comparison
across replicates, and power-response tab.

All figure *construction* comes from core/plotting.py; this window's job is
only to fetch the right data, call those builders, and embed the result via
FigureCanvasTkAgg. Tracks its own figures explicitly and closes only those
-- never plt.close('all'), which would also kill the main window's Lasso
canvas.
"""
import re
import gc

import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..core import persistence, pooling, plotting


class AnalyticsDashboardWindow(ctk.CTkToplevel):
    def __init__(self, master_app):
        super().__init__(master_app)
        self.master_app = master_app

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

    def _get_pooled_data(self):
        df = persistence.read_summary_csv(self.master_app.master_folder_path)
        return pooling.pool_replicates(df)

    def refresh_views(self):
        self.render_current_fits()
        self.render_global_summary()
        self.render_power_response()
        # See thread_bridge.py's docstring for why this matters: reclaim the
        # figures just closed above, deterministically, on the main thread.
        gc.collect()

    def render_current_fits(self):
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
        fig_fit = plotting.build_current_fit_figure(traces)

        self.current_fit_fig = fig_fit
        canvas_fit = FigureCanvasTkAgg(fig_fit, master=self.tab_current)
        canvas_fit.get_tk_widget().pack(fill="both", expand=True)
        canvas_fit.draw()

    def render_global_summary(self):
        if self.global_fig is not None:
            try:
                plt.close(self.global_fig)
            except Exception:
                pass
            self.global_fig = None

        for widget in self.tab_global.winfo_children():
            widget.destroy()

        grouped_df, anomaly_text = self._get_pooled_data()
        self.master_app.update_metadata_box_display(anomaly_text)

        fig_global = plotting.build_global_summary_figure(grouped_df, dark_mode=True)
        if fig_global is None:
            lbl = ctk.CTkLabel(self.tab_global, text="No pooled replicate data found in summary CSV.", font=ctk.CTkFont(slant="italic"))
            lbl.pack(expand=True)
            return

        self.global_fig = fig_global
        canvas_global = FigureCanvasTkAgg(fig_global, master=self.tab_global)
        canvas_global.get_tk_widget().pack(fill="both", expand=True)
        canvas_global.draw()

    def render_power_response(self):
        if self.power_fig is not None:
            try:
                plt.close(self.power_fig)
            except Exception:
                pass
            self.power_fig = None

        for widget in self.tab_power.winfo_children():
            widget.destroy()

        grouped_df, _ = self._get_pooled_data()
        wavelengths = pooling.get_available_wavelengths(grouped_df)
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

        grouped_df, _ = self._get_pooled_data()
        fig = plotting.build_power_response_figure(grouped_df, wl_val, dark_mode=True)
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
