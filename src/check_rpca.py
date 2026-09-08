import pandas as pd

old = pd.read_csv(r"C:\Projects\thesis\data\PULSE_files_SQI\summaries\summary_20260622_160818.csv")
new = pd.read_csv(r"C:\Projects\thesis\data\PULSE_files_RPCA\summaries\summary_RPCA_20260625_234941.csv")

# Merge on rec_id and compare bpm_smoothed
merged = old[['rec_id', 'bpm_smoothed', 'err_smoothed']].merge(
    new[['rec_id', 'bpm_smoothed', 'err_smoothed']],
    on='rec_id', suffixes=('_old', '_new'))

merged['bpm_diff'] = (merged['bpm_smoothed_old'] - merged['bpm_smoothed_new']).abs()
merged['err_diff'] = (merged['err_smoothed_old'] - merged['err_smoothed_new']).abs()

print("Recordings where RPCA differs from locked production:")
print(merged[merged['bpm_diff'] > 0.5].sort_values('bpm_diff', ascending=False))

print(f"\nMax BPM diff: {merged['bpm_diff'].max():.2f}")
print(f"Mean BPM diff: {merged['bpm_diff'].mean():.3f}")