import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json

# --- 1. ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ (ΑΠΟ ΤΑ JSON ΣΟΥ) ---
PULSE_DIR = Path(r"C:\Projects\thesis\data\PULSE_files_SQI")
data_list = []

for json_path in PULSE_DIR.glob("selection_windowed_*.json"):
    with open(json_path, 'r') as f:
        data = json.load(f)
        # Από το όνομα του αρχείου παίρνουμε τα χαρακτηριστικά
        rec_id = json_path.stem.replace("selection_windowed_", "")
        parts = rec_id.split('_')
        subject = parts[0]
        condition = "_".join(parts[1:])
        
        data_list.append({
            'Subject': subject,
            'Condition': condition,
            'MAE': data.get("error_bpm")
        })

df = pd.DataFrame(data_list)

# --- 2. ΣΤΥΛ ΚΑΙ ΧΡΩΜΑΤΑ (ΣΥΝΕΠΕΙΑ) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11
})

# Χρησιμοποιούμε την παλέτα: Μπλε (καλό) -> Γκρι -> Κόκκινο (κακό)
colors = ["#5b7c99", "#f5f5f5", "#c44e52"] 
cmap = LinearSegmentedColormap.from_list("custom_mae", colors, N=256)

# --- 3. HEATMAP ---
plt.figure(figsize=(11, 5))
pivot_df = df.pivot(index='Condition', columns='Subject', values='MAE')
sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap=cmap, vmin=0, vmax=12, linewidths=0.5, linecolor='white')
plt.title('Per-recording MAE (Heatmap)')
plt.tight_layout()
plt.savefig(Path(r"C:\Projects\thesis\thesis_latex\images") / 'heatmap_mae.png', dpi=300)
plt.close()

# --- 4. BOXPLOT ---
plt.figure(figsize=(11, 5))
# Ταξινόμηση υποκειμένων βάσει του μέσου MAE
order = df.groupby('Subject')['MAE'].mean().sort_values().index
sns.boxplot(x='Subject', y='MAE', data=df, color="#5b7c99", order=order, showfliers=True)
sns.stripplot(x='Subject', y='MAE', data=df, color='black', alpha=0.3, order=order)
plt.title('Distribution of Absolute Error per Subject (Boxplot)')
plt.ylabel('Absolute Error (BPM)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(Path(r"C:\Projects\thesis\thesis_latex\images") / 'boxplot_mae.png', dpi=300)
plt.close()

print("✅ Τα plots αποθηκεύτηκαν επιτυχώς!")