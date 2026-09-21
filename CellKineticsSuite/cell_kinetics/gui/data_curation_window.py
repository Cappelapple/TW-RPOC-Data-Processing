"""Popup window for including/excluding individual processed datasets from
the pooled mean/std -- e.g. a trial with a bad segmentation or an outlier
control that's inflating the error bars on the summary plots.

Exclusions are written back into the summary CSV's own "Excluded" column
immediately on toggle (persistence.write_summary_csv), so they survive an
app relaunch and stay in effect until Start New Session archives that CSV
away and starts a fresh one -- there is deliberately no separate
session-only mode.

Outlier flagging (pooling.leave_one_out_outlier_flags) is recomputed after
every toggle, not just once at load -- excluding the current worst offender
in a group can reveal the next one, so the flags should track whichever
replicates are still included right now, not a static first-pass snapshot.
"""
import pandas as pd
import tkinter as tk
import customtkinter as ctk

from ..core import persistence, pooling

OUTLIER_FLAG_COLOR = "#E67E22"


class DataCurationWindow(ctk.CTkToplevel):
    def __init__(self, master_app):
        super().__init__(master_app)
        self.master_app = master_app

        self.title("Manage Data — Include/Exclude Replicates")
        self.geometry("860x680")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        top_bar = ctk.CTkFrame(self, fg_color="transparent")
        top_bar.pack(fill="x", padx=10, pady=(10, 0))
        ctk.CTkLabel(
            top_bar,
            text="Toggle a replicate off to exclude it from the pooled mean/std shown in the "
                 "Analytics window. Saved immediately to the summary CSV. Orange = excluding it "
                 "would shrink this group's spread the most (only shown once a group's spread "
                 "exceeds the same threshold used for the app's own high-spread warning).",
            font=ctk.CTkFont(size=11, slant="italic"), wraplength=740, justify="left",
        ).pack(side="left")
        ctk.CTkButton(top_bar, text="↻ Reload", width=90, command=self.reload).pack(side="right")

        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)

        self.lbl_status = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=10, slant="italic"))
        self.lbl_status.pack(pady=(0, 8))

        self.df = None
        self.row_vars = {}
        self.checkboxes = {}
        self.base_texts = {}
        self.default_text_color = None
        self.reload()

    def reload(self):
        for child in self.scroll.winfo_children():
            child.destroy()
        self.row_vars = {}
        self.checkboxes = {}
        self.base_texts = {}

        df = persistence.read_summary_csv(self.master_app.master_folder_path)
        if df is None or df.empty:
            ctk.CTkLabel(self.scroll, text="No summary data yet -- process at least one dataset first.").pack(pady=20)
            self.df = None
            return

        if "Excluded" not in df.columns:
            df["Excluded"] = False
        df["Excluded"] = df["Excluded"].fillna(False).astype(bool)
        df["Wavelength_Num"] = df["Wavelength"].astype(str).str.extract(r"(\d+)").astype(float)
        self.df = df.reset_index(drop=True)

        group_cols = ["Wavelength_Num", "Power_mW", "Condition", "Fluorophore"]
        for _, group_index in self.df.groupby(group_cols, dropna=False, sort=True).groups.items():
            self._build_group(list(group_index))

        self.lbl_status.configure(text="")

    def _build_group(self, row_indices):
        first = self.df.loc[row_indices[0]]
        group_frame = ctk.CTkFrame(self.scroll, fg_color="#1E1E1E", corner_radius=6)
        group_frame.pack(fill="x", pady=(8, 2), padx=4)

        header = ctk.CTkFrame(group_frame, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(
            header,
            text=f"{first['Fluorophore']}  |  {first['Wavelength']}  {first['Power_mW']}mW  {first['Condition']}",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="left")
        stats_lbl = ctk.CTkLabel(header, text="", font=ctk.CTkFont(size=11), text_color="#A9DFBF")
        stats_lbl.pack(side="right")

        rows_frame = ctk.CTkFrame(group_frame, fg_color="transparent")
        rows_frame.pack(fill="x", padx=16, pady=(0, 6))

        def refresh_group():
            included = [i for i in row_indices if self.row_vars[i].get()]
            ks = self.df.loc[included, "Decay_Constant_k"].dropna()

            if len(ks) == 0:
                stats_lbl.configure(text="No included replicates")
            else:
                mean_k = ks.mean()
                std_k = ks.std() if len(ks) > 1 else 0.0
                flag = "  ⚠ High spread" if std_k > pooling.STD_K_OUTLIER_THRESHOLD else ""
                stats_lbl.configure(text=f"Mean k = {mean_k:.4f} ± {std_k:.4f}  (N={len(ks)}){flag}")

            outlier_flags = pooling.leave_one_out_outlier_flags(
                {i: self.df.loc[i, "Decay_Constant_k"] for i in included}
            )
            for i in row_indices:
                chk = self.checkboxes[i]
                if i in outlier_flags and self.row_vars[i].get():
                    current_std, new_std = outlier_flags[i]
                    chk.configure(
                        text=f"{self.base_texts[i]}    ⚠ excl. → σ {current_std:.4f}→{new_std:.4f}",
                        text_color=OUTLIER_FLAG_COLOR,
                    )
                else:
                    chk.configure(text=self.base_texts[i], text_color=self.default_text_color)

        for idx in row_indices:
            row = self.df.loc[idx]
            var = tk.BooleanVar(value=not bool(row["Excluded"]))
            self.row_vars[idx] = var
            k_val = row["Decay_Constant_k"]
            k_text = f"{k_val:.4f}" if pd.notna(k_val) else "NaN"
            base_text = f"{row['Dataset']}    k = {k_text}    ({row.get('Fit_Model', 'Single-Exponential')})"
            self.base_texts[idx] = base_text

            chk = ctk.CTkCheckBox(
                rows_frame, text=base_text, variable=var,
                command=lambda idx=idx, var=var, refresh_group=refresh_group: self._on_toggle(idx, var, refresh_group),
            )
            chk.pack(anchor="w", pady=1)
            self.checkboxes[idx] = chk
            if self.default_text_color is None:
                self.default_text_color = chk.cget("text_color")

        refresh_group()

    def _on_toggle(self, idx, var, refresh_group):
        self.df.loc[idx, "Excluded"] = not var.get()
        refresh_group()
        self._save()

    def _save(self):
        try:
            save_df = self.df.drop(columns=["Wavelength_Num"], errors="ignore")
            persistence.write_summary_csv(self.master_app.master_folder_path, save_df)
            self.lbl_status.configure(text="Saved.")
        except Exception as e:
            self.lbl_status.configure(text=f"Save failed: {e}")
        self.master_app.update_analytics_window_if_open()

    def on_close(self):
        self.master_app.data_curation_window = None
        self.destroy()
