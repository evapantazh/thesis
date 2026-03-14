import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load grid data
OUTPUT_CSV = r"C:\Projects\thesis\data\csv\chrys_500_tshirt_GRID.csv"

df = pd.read_csv(OUTPUT_CSV)

# Convert frames to seconds
df['seconds'] = df['frame'] / 30.0

# --- OPTION 1: Plot all 28 cells ---
plt.figure(figsize=(16, 10))

# Create 4x7 grid of subplots (matching your chest grid)
for cell_id in range(28):
    row = cell_id // 4  # 7 rows
    col = cell_id % 4   # 4 columns
    
    plt.subplot(7, 4, cell_id + 1)
    plt.plot(df['seconds'], df[f'cell_{cell_id}'], linewidth=0.5)
    plt.title(f'Cell {cell_id}', fontsize=8)
    plt.ylabel('Depth (mm)', fontsize=6)
    if row == 6:  # Only bottom row
        plt.xlabel('Time (s)', fontsize=6)
    plt.grid(True, alpha=0.3)
    plt.tick_params(labelsize=6)

plt.tight_layout()
plt.suptitle('4x7 Grid: All Chest Regions', fontsize=14, y=1.00)
plt.show()

# --- OPTION 2: Compare center vs edges ---
# Center cells might have stronger cardiac signal
center_cells = [9, 10, 13, 14]  # Middle 4 cells
edge_cells = [0, 3, 24, 27]      # Corner cells

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
for cell in center_cells:
    plt.plot(df['seconds'], df[f'cell_{cell}'], label=f'Cell {cell}', linewidth=1)
plt.title('Center Chest Cells (Expected: Stronger Heart Signal)')
plt.ylabel('Depth (mm)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
for cell in edge_cells:
    plt.plot(df['seconds'], df[f'cell_{cell}'], label=f'Cell {cell}', linewidth=1, alpha=0.7)
plt.title('Edge Chest Cells (Expected: Weaker Signal)')
plt.xlabel('Time (seconds)')
plt.ylabel('Depth (mm)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- OPTION 3: Mean of all cells (similar to your old plot) ---
cell_columns = [f'cell_{i}' for i in range(28)]
df['mean_depth'] = df[cell_columns].mean(axis=1)

plt.figure(figsize=(12, 6))
plt.plot(df['seconds'], df['mean_depth'], color='blue', linewidth=1)
plt.title('Mean Depth Across All 28 Grid Cells')
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Depth (mm)')
plt.grid(True, alpha=0.3)
plt.show()