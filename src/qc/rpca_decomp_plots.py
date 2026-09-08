#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# --- 1. SETTINGS, PATHS & STYLE ---
RECORDINGS = [
    "AGA_800_hoodie",
    "GAX_800_tshirt",
    "GAX_1200_hoodie",
    "GPA_800_hoodie"
]

FILTERED_DIR = Path(r"C:\Projects\thesis\data\FILTERED_files")
# Απευθείας φάκελος μόνο για τα plots
PLOT_DIR = Path(r"C:\Projects\thesis\thesis_latex\images")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Thesis Academic Styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 8, 'legend.fontsize': 8.5,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

# Academic color palette
C_D = "#5b7c99"      # Muted Blue
C_X = "#55a868"      # Muted Green
C_S = "#c44e52"      # Muted Red

# RPCA parameter
GAMMA = 0.03
CELL_TO_PLOT = 14


# --- 2. IALM ALGORITHM ---
def fit_ialm(D, lambda_=None, tol=1e-7, mu=None, rho=1.5, max_iter=1000, verbose=False):
    """
    Inexact Augmented Lagrange Multiplier method for RPCA.
    """
    D = np.asarray(D, dtype=np.float64)
    m, n = D.shape
    if lambda_ is None:
        lambda_ = 1 / np.sqrt(m)

    Y = D.copy()
    norm_two = np.linalg.norm(Y, 2)
    norm_inf = np.linalg.norm(Y, np.inf) / lambda_
    dual_norm = max(norm_two, norm_inf)
    Y /= dual_norm

    A = np.zeros_like(D)
    E = np.zeros_like(D)

    if mu is None:
        mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    d_norm = np.linalg.norm(D, 'fro')

    for it in range(1, max_iter + 1):
        temp_T = D - A + (1 / mu) * Y
        E = np.maximum(temp_T - lambda_ / mu, 0) + np.minimum(temp_T + lambda_ / mu, 0)

        U, S, Vt = np.linalg.svd(D - E + (1 / mu) * Y, full_matrices=False)
        svp = np.sum(S > 1 / mu)
        if svp == 0:
            A = np.zeros_like(D)
        else:
            A = (U[:, :svp] * (S[:svp] - 1 / mu)) @ Vt[:svp, :]

        Z = D - A - E
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)
        stop = np.linalg.norm(Z, 'fro') / d_norm

        if stop < tol:
            break
    return A, E


# --- 3. BATCH EXECUTION & PLOTTING ---
for REC_ID in RECORDINGS:
    print(f"\nProcessing {REC_ID}...")
    INPUT_PATH = FILTERED_DIR / f"FILTERED_{REC_ID}.csv"
    
    if not INPUT_PATH.exists():
        print(f"❌ Error: {INPUT_PATH} not found. Skipping.")
        continue

    # Load data
    df = pd.read_csv(INPUT_PATH)
    frames = df['frame'].values
    D = df.drop(columns=['frame']).values

    # Run RPCA
    X_low_rank, S_sparse = fit_ialm(D, lambda_=GAMMA, verbose=False)

    # Fetch FS for time axis
    meta_file = Path(r"C:\Projects\thesis\data\metadata") / f"meta_{REC_ID}.json"
    try:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        FS = float(meta.get("actual_fps", 15.0))
    except FileNotFoundError:
        FS = 15.0
    time_axis = np.arange(len(frames)) / FS

    # --- VISUALIZATION ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    
    # D: Filtered Input
    axes[0].plot(time_axis, D[:, CELL_TO_PLOT], linewidth=0.8, color=C_D)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("D — Filtered input (cardiac + breathing + noise)")

    # X: Low-rank
    axes[1].plot(time_axis, X_low_rank[:, CELL_TO_PLOT], linewidth=0.8, color=C_X)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("X — Low-rank (cardiac + breathing)")

    # S: Sparse
    axes[2].plot(time_axis, S_sparse[:, CELL_TO_PLOT], linewidth=0.8, color=C_S)
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("S — Sparse (motion artifacts, outliers)")

    # Clean up spines for academic look
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.01)

    fig.subplots_adjust(hspace=0.35)
    fig.tight_layout()

    # Save
    plot_path = PLOT_DIR / f"RPCA_{REC_ID}_cell_{CELL_TO_PLOT}.pdf"
    fig.savefig(plot_path)
    fig.savefig(plot_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    
    print(f"✅ Saved plot: {plot_path.name}")

print("\n🎉 All 4 plots generated successfully!")