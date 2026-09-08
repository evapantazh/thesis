import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from pathlib import Path  # <-- Διόρθωση του import

# Ορίζουμε ΠΟΥ θέλουμε να πάει η εικόνα
IMAGE_DIR = Path(r"C:\Projects\thesis\thesis_latex\images") 
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
final_plot_path = IMAGE_DIR / 'performance_heatmap.png'

# --- 1. ΤΑ ΠΡΑΓΜΑΤΙΚΑ ΣΟΥ ΔΕΔΟΜΕΝΑ ---
data_string = """recording_id,MAE
AGA_1200_hoodie,4.8
AGA_1200_tshirt,2.5
AGA_1800_hoodie,12.6
AGA_1800_tshirt,5.9
AGA_800_hoodie,4.2
AGA_800_tshirt,2.2
AGE_1200_hoodie,2.2
AGE_1200_tshirt,2.5
AGE_1800_hoodie,4.9
AGE_1800_tshirt,5.3
AGE_800_hoodie,10.0
AGE_800_tshirt,18.6
AVE_1200_hoodie,5.1
AVE_1200_tshirt,2.0
AVE_1800_hoodie,6.8
AVE_1800_tshirt,6.5
AVE_800_hoodie,4.0
AVE_800_tshirt,2.3
EST_1200_hoodie,6.0
EST_1200_tshirt,6.4
EST_1800_hoodie,4.1
EST_1800_tshirt,13.3
EST_800_hoodie,9.7
GAX_1200_hoodie,4.2
GAX_1200_tshirt,2.8
GAX_1800_hoodie,7.3
GAX_1800_tshirt,3.6
GAX_800_hoodie,9.6
GAX_800_tshirt,5.4
GBA_1200_hoodie,7.0
GBA_1200_tshirt,8.8
GBA_1800_hoodie,5.1
GBA_1800_tshirt,4.6
GBA_800_hoodie,4.0
GBA_800_tshirt,5.9
GPA_1200_hoodie,4.4
GPA_1200_tshirt,0.9
GPA_1800_hoodie,1.0
GPA_1800_tshirt,4.5
GPA_800_hoodie,1.7
GPA_800_tshirt,0.5
GVA_1200_hoodie,3.9
GVA_1200_tshirt,3.1
GVA_1800_hoodie,5.2
GVA_1800_tshirt,5.1
GVA_800_hoodie,4.2
GVA_800_tshirt,0.7
IMA_1200_hoodie,1.3
IMA_1200_tshirt,3.3
IMA_1800_hoodie,0.4
IMA_1800_tshirt,9.8
IMA_800_hoodie,13.8
IMA_800_tshirt,1.7
KGI_1200_hoodie,8.7
KGI_1200_tshirt,4.8
KGI_1800_hoodie,4.2
KGI_1800_tshirt,8.0
KGI_800_hoodie,9.8
KGI_800_tshirt,15.3
MGK_1200_hoodie,0.2
MGK_1200_tshirt,0.4
MGK_1800_tshirt,0.5
MGK_800_hoodie,1.0
MGK_800_tshirt,2.3
MPA_1200_hoodie,7.1
MPA_1200_tshirt,3.7
MPA_1800_hoodie,5.0
MPA_1800_tshirt,2.5
MPA_800_hoodie,3.2
MPA_800_tshirt,5.0
MST_1200_hoodie,7.5
MST_1200_tshirt,3.6
MST_1800_hoodie,4.6
MST_1800_tshirt,5.1
MST_800_hoodie,3.5
MST_800_tshirt,1.4
"""

df = pd.read_csv(io.StringIO(data_string))
df['Subject'] = df['recording_id'].apply(lambda x: x.split('_')[0])
df['Condition'] = df['recording_id'].apply(lambda x: '_'.join(x.split('_')[1:]))

subjects_order = ['MGK', 'GPA', 'GVA', 'MST', 'MPA', 'AVE', 'IMA', 'AGA', 'GAX', 'GBA', 'AGE', 'EST', 'KGI']
conditions_order = ['800_tshirt', '800_hoodie', '1200_tshirt', '1200_hoodie', '1800_tshirt', '1800_hoodie']

heatmap_data = df.pivot(index='Condition', columns='Subject', values='MAE')
heatmap_data = heatmap_data.reindex(index=conditions_order, columns=subjects_order)


# --- 2. ΣΤΥΛ ΚΑΙ ΧΡΩΜΑΤΑ ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Linux Libertine', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11
})

# RPCA Colors: Μπλε -> Ανοιχτό Γκρι -> Κόκκινο
colors = ["#5b7c99", "#f5f5f5", "#c44e52"] 
cmap = LinearSegmentedColormap.from_list("custom_mae", colors, N=256)
cmap.set_bad(color='#e0e0e0')


# --- 3. ΣΧΕΔΙΑΣΜΟΣ HEATMAP ---
fig, ax = plt.subplots(figsize=(11, 5))

sns.heatmap(heatmap_data, 
            annot=True,        
            fmt=".1f",         
            cmap=cmap,         
            vmin=0, vmax=12,   
            linewidths=0.5,    
            linecolor='white',
            cbar_kws={'label': 'Mean Absolute Error (BPM)'},
            ax=ax)

ax.set_xlabel('Subjects (Ordered by Increasing Mean HR $\\rightarrow$)', fontsize=12, labelpad=10)
ax.set_ylabel('Acquisition Condition', fontsize=12, labelpad=10)
ax.set_title('Per-recording MAE across all Subjects and Conditions', pad=15)

plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()

# --- 4. ΣΩΣΤΗ ΑΠΟΘΗΚΕΥΣΗ ---
plt.savefig(final_plot_path, dpi=300) # ΕΔΩ γίνεται η αποθήκευση!
plt.show() # Εμφάνιση του σωστού παραθύρου
print(f"✅ Το Heatmap αποθηκεύτηκε επιτυχώς στον φάκελο:\n{final_plot_path}")