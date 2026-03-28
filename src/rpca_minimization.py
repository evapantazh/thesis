import numpy as np
import pandas as pd
import os

# --- THE IALM ENGINE ---


def fit_ialm(D, lambda_=None, tol=1e-7, mu=None, rho=1.5, max_iter=1000, verbose=True):
    D = np.asarray(D, dtype=np.float64)
    m, n = D.shape

    if lambda_ is None:
        lambda_ = 1 / np.sqrt(m)

    # Init Y (scaled)
    Y = D.copy()
    norm_two = np.linalg.norm(Y, 2)
    norm_inf = np.linalg.norm(Y, np.inf) / lambda_
    dual_norm = max(norm_two, norm_inf)
    Y /= dual_norm

    A = np.zeros_like(D)
    E = np.zeros_like(D)

    if mu is None:
        mu = 1.25 / norm_two
    mu_bar = mu * 1e7

    d_norm = np.linalg.norm(D, 'fro')

    for it in range(1, max_iter + 1):

        # 1) Sparse update: soft threshold
        temp_T = D - A + (1/mu) * Y
        E = np.maximum(temp_T - lambda_/mu, 0) + np.minimum(temp_T + lambda_/mu, 0)

        # 2) Low-rank update: SVT
        U, S, Vt = np.linalg.svd(D - E + (1/mu) * Y, full_matrices=False)
        svp = np.sum(S > 1/mu)
        if svp == 0:
            A = np.zeros_like(D)
        else:
            A = (U[:, :svp] * (S[:svp] - 1/mu)) @ Vt[:svp, :]

        # 3) Multiplier update
        Z = D - A - E
        Y = Y + mu * Z

        # 4) mu update (standard)
        mu = min(mu * rho, mu_bar)

        # stopping criterion
        stop = np.linalg.norm(Z, 'fro') / d_norm

        if verbose and (it % 1 == 0):
            nz_E = np.sum(np.abs(E) > 1e-12)
            print(f"Iter {it:4d} | rank(A) {np.linalg.matrix_rank(A):2d} | "
                  f"nz(E) {nz_E:7d} | mu {mu:.2e} | stop {stop:.2e}")

        if stop < tol:
            break

    return A, E

# --- DATA LOADING ---
file_path = r"C:\Projects\thesis\data\FILTERED_Sub01_500_Tshirt.csv"
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found!")
else:
    df = pd.read_csv(file_path)
    frames = df['frame'].values
    D_matrix = df.drop(columns=['frame']).values

    print(f"Matrix loaded. Shape: {D_matrix.shape}")

    # --- RUN THE ENGINE ---
    A_heart, E_noise = fit_ialm(D_matrix, verbose=True)

    # --- SAVE THE RESULTS ---
    df_low_rank = pd.DataFrame(A_heart, columns=df.columns[1:])
    df_low_rank.insert(0, 'frame', frames)
    
    output_name = "Sub01_Matrix_X_LowRank.csv"
    df_low_rank.to_csv(output_name, index=False)
    print(f"\n✅ SUCCESS! File saved: {output_name}")