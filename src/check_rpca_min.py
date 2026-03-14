import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Use the full paths to avoid the FileNotFoundError
path_orig = r"C:\Projects\thesis\data\csv\Sub01_Matrix_D_Filtered.csv"
path_clean = r"C:\Projects\thesis\Sub01_Matrix_X_LowRank.csv"

try:
    # Load Original and Cleaned data
    df_orig = pd.read_csv(path_orig)
    df_clean = pd.read_csv(path_clean)

    cell = "cell_14" # The center of the chest

    plt.figure(figsize=(12, 8))

    # Plot 1: The Signal Cleanup
    # We look at the first 300 frames (10 seconds at 30fps)
    plt.subplot(2, 1, 1)
    plt.plot(df_orig[cell][:300], label="Raw Filtered (D)", color="gray", alpha=0.4)
    plt.plot(df_clean[cell][:300], label="RPCA Cleaned (X)", color="red", linewidth=1.5)
    plt.title(f"Verification: Signal Cleanup for {cell}")
    plt.xlabel("Frames")
    plt.ylabel("Displacement")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: The Singular Values (The 'Pattern' Check)
    # This is the "Nuclear Norm" visual proof
    u, s_orig, vh = np.linalg.svd(df_orig.drop(columns=['frame']).values, full_matrices=False)
    u, s_clean, vh = np.linalg.svd(df_clean.drop(columns=['frame']).values, full_matrices=False)

    plt.subplot(2, 1, 2)
    plt.plot(s_orig[:20], 'o-', label="Original Singular Values", color="blue", markersize=4)
    plt.plot(s_clean[:20], 'x-', label="RPCA Singular Values", color="green", markersize=6)
    plt.title("Verification: Nuclear Norm Effect (Low-Rank Extraction)")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Magnitude (Log Scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

except FileNotFoundError as e:
    print(f"Error: Could not find the file. Check path: {e}")