import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# --- 1. SETTINGS & PATHS ---
SUBJECT = "MST"
DIST = "800"
CLOTH = "tshirt"   # tshirt    hoodie
REC_ID = f"{SUBJECT}_{DIST}_{CLOTH}"

FILTERED_DIR = Path(r"C:\Projects\thesis\data\FILTERED_files")
OUTPUT_DIR = Path(r"C:\Projects\thesis\data\RPCA_files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = FILTERED_DIR / f"FILTERED_{REC_ID}.csv"
OUTPUT_LOW_RANK = OUTPUT_DIR / f"LOWRANK_{REC_ID}.csv"
OUTPUT_SPARSE = OUTPUT_DIR / f"SPARSE_{REC_ID}.csv"

# RPCA parameter from paper (Section C)
GAMMA = 0.02
# Alternative: default from IALM paper = 1/sqrt(m), computed after loading data
#GAMMA = 1 / np.sqrt(num_frames)  # uncomment and move after data loading
#GAMMA = 1 / np.sqrt(1101)  # ≈ 0.0301


# --- 2. IALM ALGORITHM ---
def fit_ialm(D, lambda_=None, tol=1e-7, mu=None, rho=1.5, max_iter=1000, verbose=True):
    """
    Inexact Augmented Lagrange Multiplier method for RPCA.
    Decomposes D = A + E where A is low-rank, E is sparse.
    Reference: Lin, Chen, Ma (2010) - Algorithm 5.
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
        # Sparse update: soft thresholding
        temp_T = D - A + (1 / mu) * Y
        E = np.maximum(temp_T - lambda_ / mu, 0) + np.minimum(temp_T + lambda_ / mu, 0)

        # Low-rank update: SVD thresholding
        U, S, Vt = np.linalg.svd(D - E + (1 / mu) * Y, full_matrices=False)
        svp = np.sum(S > 1 / mu)
        if svp == 0:
            A = np.zeros_like(D)
        else:
            A = (U[:, :svp] * (S[:svp] - 1 / mu)) @ Vt[:svp, :]

        # Multiplier update
        Z = D - A - E
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        # Stopping criterion
        stop = np.linalg.norm(Z, 'fro') / d_norm

        if verbose and (it % 10 == 0 or stop < tol):
            print(f"  Iter {it:4d} | rank(A)={np.linalg.matrix_rank(A):2d} | "
                  f"nnz(E)={np.sum(np.abs(E) > 1e-12):7d} | stop={stop:.2e}")

        if stop < tol:
            print(f"  Converged at iteration {it}")
            break

    return A, E


# --- 3. EXECUTION ---
if not INPUT_PATH.exists():
    print(f"❌ Error: {INPUT_PATH} not found!")
else:
    df = pd.read_csv(INPUT_PATH)
    frames = df['frame'].values
    D = df.drop(columns=['frame']).values

    print(f"Input matrix D: {D.shape} (frames × cells)")
    print(f"Using gamma = {GAMMA}")

    # Run RPCA
    X_low_rank, S_sparse = fit_ialm(D, lambda_=GAMMA, verbose=True)

    # Save low-rank matrix
    df_X = pd.DataFrame(X_low_rank, columns=[f"cell_{i}" for i in range(28)])
    df_X.insert(0, 'frame', frames)
    df_X.to_csv(OUTPUT_LOW_RANK, index=False)

    # Save sparse matrix
    df_S = pd.DataFrame(S_sparse, columns=[f"cell_{i}" for i in range(28)])
    df_S.insert(0, 'frame', frames)
    df_S.to_csv(OUTPUT_SPARSE, index=False)

    print(f"\n✅ Low-rank saved: {OUTPUT_LOW_RANK}")
    print(f"✅ Sparse saved: {OUTPUT_SPARSE}")
    print(f"   rank(X) = {np.linalg.matrix_rank(X_low_rank)}")
    print(f"   nnz(S)  = {np.sum(np.abs(S_sparse) > 1e-12)} / {S_sparse.size}")

    # --- 4. VISUAL CHECK ---
    CELL_TO_PLOT = 14
    cell_name = f"cell_{CELL_TO_PLOT}"

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"RPCA Decomposition — {REC_ID} — {cell_name}", fontsize=14)

    time_axis = np.arange(len(frames)) / 15.0  # approximate

    # Original filtered signal
    axes[0].plot(time_axis, D[:, CELL_TO_PLOT], linewidth=0.6)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("D — Filtered input (cardiac + breathing + noise)")

    # Low-rank component
    axes[1].plot(time_axis, X_low_rank[:, CELL_TO_PLOT], linewidth=0.6, color="tab:green")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("X — Low-rank (cardiac + breathing)")

    # Sparse component
    axes[2].plot(time_axis, S_sparse[:, CELL_TO_PLOT], linewidth=0.6, color="tab:red")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("S — Sparse (motion artifacts, outliers)")

    plt.tight_layout()

    PLOT_DIR = OUTPUT_DIR / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOT_DIR / f"rpca_check_{REC_ID}_{cell_name}.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"📊 Plot saved: {plot_path}")