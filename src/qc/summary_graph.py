import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Φόρτωση δεδομένων
df = pd.read_csv(Path(r"C:\Projects\thesis\final_results_summary.csv"))
df = df.dropna(subset=['Error']) # Καθαρισμός αν υπάρχουν κενά

# ---------------------------------------------------------
# SLIDE 1: Histogram (Κατανομή Σφαλμάτων)
# ---------------------------------------------------------
bins = [0, 5, 10, 15, 100]
labels = ['0-5 BPM', '5-10 BPM', '10-15 BPM', '>15 BPM']
df['Category'] = pd.cut(df['Error'], bins=bins, labels=labels, right=True)
hist_data = df['Category'].value_counts().reindex(labels)

plt.figure(figsize=(8, 6))
bars = plt.bar(hist_data.index, hist_data.values, color=['green', 'yellow', 'orange', 'red'])
plt.title('Distribution of Heart Rate Estimation Errors')
plt.ylabel('Number of Recordings')
plt.xlabel('Error Range (BPM)')
plt.grid(axis='y', alpha=0.3)
plt.savefig(r"C:\Projects\thesis\histogram_error.png", dpi=300)
print("Histogram saved.")

# ---------------------------------------------------------
# SLIDE 2: Comparison Chart (5 Καλύτερα vs 5 Χειρότερα)
# ---------------------------------------------------------
sorted_df = df.sort_values('Error')
best_worst = pd.concat([sorted_df.head(5), sorted_df.tail(5)])

plt.figure(figsize=(10, 6))
colors = ['green'] * 5 + ['red'] * 5
plt.bar(best_worst['Recording'], best_worst['Error'], color=colors)
plt.xticks(rotation=45, ha='right')
plt.title('Comparison: 5 Best vs 5 Worst Performing Recordings')
plt.ylabel('Error (BPM)')
plt.axhline(y=5, color='gray', linestyle='--', linewidth=0.8, label='Error Threshold (5 BPM)')
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Projects\thesis\best_worst_comparison.png", dpi=300)
print("Comparison chart saved.")