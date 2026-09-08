import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec

# ============================================================
# 1. UNIFORM THESIS STYLING BLOCK
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 13,       # Slightly larger for matrix labels (M, L, S)
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ============================================================
# RPCA Inexact ALM Function
# ============================================================
def fit_ialm(D, lambda_=None, tol=1e-6, mu=None, rho=1.5, max_iter=200):
    D = np.asarray(D, dtype=np.float64)
    m, n = D.shape
    if lambda_ is None:
        lambda_ = 1 / np.sqrt(m)

    Y = D.copy()
    norm_two = np.linalg.norm(Y, 2)
    norm_inf = np.linalg.norm(Y, np.inf) / lambda_
    Y /= max(norm_two, norm_inf)

    A = np.zeros_like(D)
    E = np.zeros_like(D)

    if mu is None:
        mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    d_norm = np.linalg.norm(D, 'fro')

    for _ in range(1, max_iter + 1):
        temp_T = D - A + (1 / mu) * Y
        E = np.maximum(temp_T - lambda_ / mu, 0) + np.minimum(temp_T + lambda_ / mu, 0)

        U, S, Vt = np.linalg.svd(D - E + (1 / mu) * Y, full_matrices=False)
        svp = int(np.sum(S > 1 / mu))
        if svp == 0:
            A = np.zeros_like(D)
        else:
            A = (U[:, :svp] * (S[:svp] - 1 / mu)) @ Vt[:svp, :]

        Z = D - A - E
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        if np.linalg.norm(Z, 'fro') / d_norm < tol:
            break
    return A, E

# ============================================================
# CONFIGURATION
# ============================================================
REC_ID = "GBA_1800_hoodie"
FS = 15.0
WINDOW_SEC = 15.0
WINDOW_START_SEC = 30.0

FILTERED_PATH = Path(r"C:\Projects\thesis\data\FILTERED_files") / f"FILTERED_{REC_ID}.csv"
OUTPUT_DIR    = Path(r"C:\Projects\thesis\thesis_latex\images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH   = OUTPUT_DIR / "rpca_decomposition.pdf"

# ============================================================
# LOAD WINDOW & EXECUTE
# ============================================================
df = pd.read_csv(FILTERED_PATH)
D = df.drop(columns=['frame']).values.astype(np.float64)

start_idx = int(WINDOW_START_SEC * FS)
end_idx = start_idx + int(WINDOW_SEC * FS)
D_win = D[start_idx:end_idx]

scale = float(np.linalg.norm(D_win, 2))
D_scaled = D_win / scale
L_scaled, S_scaled = fit_ialm(D_scaled, lambda_=0.1)
L = L_scaled * scale
S = S_scaled * scale

# ============================================================
# PLOT — Three side-by-side heatmaps for M = L + S
# ============================================================
fig = plt.figure(figsize=(14, 4.0))

gs = gridspec.GridSpec(
    1, 6,
    width_ratios=[10, 1.2, 10, 1.2, 10, 0.4],
    wspace=0.0,
    figure=fig
)

ax_M    = fig.add_subplot(gs[0])
ax_eq   = fig.add_subplot(gs[1])
ax_L    = fig.add_subplot(gs[2])
ax_plus = fig.add_subplot(gs[3])
ax_S    = fig.add_subplot(gs[4])
ax_cbar = fig.add_subplot(gs[5])

# Hide gap axes
for ax_gap in (ax_eq, ax_plus):
    ax_gap.set_xticks([])
    ax_gap.set_yticks([])
    for spine in ax_gap.spines.values():
        spine.set_visible(False)
    ax_gap.set_facecolor('none')

vmax = max(abs(D_win.min()), abs(D_win.max()))
vmin = -vmax

# Mathtext configuration matching standard LaTeX notation
panels = [
    (ax_M, D_win, r'$\mathbf{M}$ (Observed)', True),
    (ax_L, L,     r'$\mathbf{L}$ (Low-rank)', False),
    (ax_S, S,     r'$\mathbf{S}$ (Sparse)',   False)
]

for ax, mat, title, show_ylabel in panels:
    im = ax.imshow(mat.T, aspect='auto', cmap='RdBu_r',
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Frame Index')
    ax.set_yticks([0, 7, 14, 21, 27])
    if show_ylabel:
        ax.set_ylabel('Cell Index')
    else:
        ax.set_yticklabels([])

# Place Operator elements inside the grid gaps with uniform serif styles
ax_eq.text(0.5, 0.5, '=', fontsize=28, ha='center', va='center', transform=ax_eq.transAxes)
ax_plus.text(0.5, 0.5, '+', fontsize=28, ha='center', va='center', transform=ax_plus.transAxes)

# Colorbar adjustments
cbar = fig.colorbar(im, cax=ax_cbar)
cbar.set_label('Amplitude (mm)', fontsize=11)
cbar.ax.tick_params(labelsize=9)

# Save configuration
plt.savefig(OUTPUT_PATH, bbox_inches='tight')
plt.savefig(OUTPUT_PATH.with_suffix('.png'), bbox_inches='tight', dpi=300)
plt.close()

print(f"✅ Polished original layout saved to: {OUTPUT_PATH}")