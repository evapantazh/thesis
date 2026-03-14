import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the CLEAN Matrix
df = pd.read_csv(r"C:\Projects\thesis\Sub01_Matrix_X_LowRank.csv")
X = df.drop(columns=['frame']).values

# 2. Get the Eigenvector weights for the heart (p2)
B = X.T @ X 
vals, vecs = np.linalg.eigh(B)
idx = vals.argsort()[::-1]
# We take the 2nd eigenvector (which gave us the 63.82 BPM)
heart_weights = np.abs(vecs[:, idx[1]]) 

# 3. Reshape the 28 weights into the 4x7 grid
grid_weights = heart_weights.reshape(7, 4)

# 4. Plot the Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(grid_weights, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("Spatial Pattern Recognition: Cardiac Signal Strength Across 4x7 Grid")
plt.xlabel("Grid Column")
plt.ylabel("Grid Row")
plt.show()