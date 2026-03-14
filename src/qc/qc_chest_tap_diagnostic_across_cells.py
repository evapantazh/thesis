"""
Tap Frame Diagnostic
Run this BEFORE 02_find_chest_tap.py to visually confirm the exact tap frame.
It shows frame-by-frame proxy values and a zoomed plot around frames 10-35.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from pathlib import Path

# ─── PATHS (same as your main script) ────────────────────────
BASE     = Path(r"C:\Projects\thesis\data")
SUBJECT  = "Sub01"
DIST     = "800"
CLOTH    = "Tshirt"
REC_ID   = f"{SUBJECT}_{DIST}_{CLOTH}"
PATH_CAMERA_GRID = BASE / f"{REC_ID}_Grid_4x7_Signal.csv"

# ─── LOAD ─────────────────────────────────────────────────────
df     = pd.read_csv(PATH_CAMERA_GRID)
frames = df["frame"].values.astype(int)
X      = df.drop(columns=["frame"]).values.astype(float)
fs     = 30.0
t      = (frames - frames[0]) / fs

# ─── REPLACE INVALID ──────────────────────────────────────────
col_medians = np.nanmedian(X, axis=0)
for j in range(X.shape[1]):
    bad = (X[:, j] == 0) | ~np.isfinite(X[:, j])
    X[bad, j] = col_medians[j]

X_dt = detrend(X, axis=0, type='linear')
dX   = np.gradient(X_dt, axis=0)

# Both proxies side by side so you can compare
energy    = np.sum(np.abs(dX), axis=1)
coherence = np.abs(np.mean(dX, axis=1))
proxy_new = energy * coherence   # coherence-weighted
proxy_old = energy               # old simple sum

# ─── PRINT TABLE: frames 10–35 ────────────────────────────────
print(f"\n{'Frame':>6}  {'t(s)':>6}  {'Energy':>12}  {'Coherence':>12}  {'Product':>12}")
print("─" * 58)

# find max in first 2 seconds to normalize for display
t2 = t <= 2.0
norm_e = proxy_old[t2].max()
norm_p = proxy_new[t2].max()

for i, (fr, ti) in enumerate(zip(frames, t)):
    if fr < 10 or fr > 40:
        continue
    bar_e = "█" * int(energy[i] / norm_e * 20)
    bar_p = "█" * int(proxy_new[i] / norm_p * 20)
    marker = " ← ?" if 16 <= fr <= 22 else ""
    print(f"  {fr:4d}  {ti:6.3f}  {energy[i]:12.1f}  {coherence[i]:12.4f}  "
          f"{proxy_new[i]:12.1f}  {bar_p}{marker}")

# ─── PLOT ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 9))
fig.suptitle(f"Tap Frame Diagnostic — {REC_ID}", fontsize=13)

zoom_mask = (t >= 0.0) & (t <= 2.0)   # first 2 seconds

# Plot 1: Old proxy (energy only)
ax = axes[0]
ax.set_title("Old proxy: sum(abs(gradient))  — frame 14 spike visible here?")
ax.plot(frames[zoom_mask], proxy_old[zoom_mask], color="steelblue")
ax.axvspan(16, 22, alpha=0.15, color="green", label="Expected tap region (fr 16–22)")
ax.set_xlabel("Frame number")
ax.set_ylabel("Energy")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Annotate top 5 peaks
top5 = np.argsort(proxy_old[zoom_mask])[-5:][::-1]
zm_frames = frames[zoom_mask]
zm_old    = proxy_old[zoom_mask]
for idx in top5:
    ax.annotate(f"fr{zm_frames[idx]}", (zm_frames[idx], zm_old[idx]),
                textcoords="offset points", xytext=(0, 6),
                fontsize=7, ha='center', color='red')

# Plot 2: New proxy (energy × coherence)
ax = axes[1]
ax.set_title("New proxy: energy × coherence  — should suppress frame 14 noise")
ax.plot(frames[zoom_mask], proxy_new[zoom_mask], color="darkorange")
ax.axvspan(16, 22, alpha=0.15, color="green", label="Expected tap region (fr 16–22)")
ax.set_xlabel("Frame number")
ax.set_ylabel("Energy × Coherence")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

zm_new = proxy_new[zoom_mask]
top5_new = np.argsort(zm_new)[-5:][::-1]
for idx in top5_new:
    ax.annotate(f"fr{zm_frames[idx]}", (zm_frames[idx], zm_new[idx]),
                textcoords="offset points", xytext=(0, 6),
                fontsize=7, ha='center', color='red')

# Plot 3: Per-cell gradient heatmap around tap region
ax = axes[2]
ax.set_title("Per-cell gradient heatmap (frames 10–35) — tap = vertical bright stripe across ALL cells")
tap_mask = (frames >= 10) & (frames <= 35)
im = ax.imshow(
    np.abs(dX[tap_mask]).T,
    aspect="auto",
    extent=[frames[tap_mask][0], frames[tap_mask][-1], 0, X.shape[1]],
    origin="lower",
    cmap="hot",
    vmax=np.percentile(np.abs(dX[tap_mask]), 99)
)
ax.axvspan(16, 22, alpha=0.3, color="cyan")
ax.set_xlabel("Frame number")
ax.set_ylabel("Cell index")
ax.set_yticks(range(0, X.shape[1], 4))
plt.colorbar(im, ax=ax, label="|gradient|")

plt.tight_layout()
out = BASE / f"tap_info/{REC_ID}_tap_diagnostic.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150)
plt.show()
print(f"\nPlot saved → {out}")
print("\nLook at Plot 3 (heatmap): the real tap = a VERTICAL bright stripe across ALL rows.")
print("That frame number is your ground truth tap frame.")