import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Common settings
# ============================================================
LINE_W = 2.0
DPI = 300
FIGSIZE = (5, 1.6)    # All panels: identical aspect ratio

# Fixed axis limits — applied to ALL panels for consistency
X_MIN, X_MAX = -0.5, 6 * np.pi + 0.5
Y_MIN, Y_MAX = -1.4, 1.4

# Time axis for sinusoids
t = np.linspace(0, 6 * np.pi, 1000)
PHASE_SHIFT = 1.5


def save_panel(filename):
    """Save current figure with consistent dimensions, no tight bbox."""
    plt.savefig(filename, dpi=DPI, transparent=True,
                bbox_inches=None, pad_inches=0)
    plt.close()


def setup_panel():
    """Create figure with consistent size and axis limits."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.axis('off')
    # Remove all margins around the axes
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


# ============================================================
# Panel 1: iToF emitted
# ============================================================
fig, ax = setup_panel()
ax.plot(t, np.sin(t), 'k', linewidth=LINE_W)
save_panel('itof_emitted.png')


# ============================================================
# Panel 2: iToF reflected
# ============================================================
fig, ax = setup_panel()
ax.plot(t, np.sin(t - PHASE_SHIFT), 'k', linewidth=LINE_W)
save_panel('itof_reflected.png')


# ============================================================
# Panel 3: dToF emitted
# ============================================================
fig, ax = setup_panel()
pulse_t = np.array([X_MIN, 3, 3, 4, 4, X_MAX])
pulse_y = np.array([0, 0, 1, 1, 0, 0])
ax.plot(pulse_t, pulse_y, 'k', linewidth=LINE_W)
save_panel('dtof_emitted.png')


# ============================================================
# Panel 4: dToF reflected
# ============================================================
fig, ax = setup_panel()
# Adjusted pulse position — closer to first pulse to make Δt
# visually comparable to Δφ (matches our earlier discussion)
pulse_t = np.array([X_MIN, 6, 6, 7, 7, X_MAX])
pulse_y = np.array([0, 0, 1, 1, 0, 0])
ax.plot(pulse_t, pulse_y, 'k', linewidth=LINE_W)
save_panel('dtof_reflected.png')

print("Done. All four PNGs have identical dimensions.")