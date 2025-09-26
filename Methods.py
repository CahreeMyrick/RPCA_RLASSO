import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, HuberRegressor, LassoCV
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.preprocessing import StandardScaler
from numpy.linalg import svd
from sklearn.metrics import mean_squared_error
import scipy.sparse.linalg as spla
import time



############################## Data Loading ##############################

def load_subj1_training_data():
    """
    Loads the training data for subject1

    Paramters
    ---------
    None

    Returns
    -------
    A 2d numpy array
    
    """
    path = '/Users/cahree/desktop/Regression/xxxxxxxxxxxx/Rapid Natural and ASL tasks/subj1/ang_vel_mat.csv'
    data = pd.read_csv(path, header=None)
    data_values = data.to_numpy()
    return data_values

def load_subj1_test_data():
    """ 
    Loads the test data for subject1

    Paramters
    ---------
    None

    Returns
    -------
    A 2d numpy array

    """
    
    path = path = '/Users/cahree/desktop/Regression/xxxxxxxxxxxxx/Rapid Natural and ASL tasks/subj1/Natural_Test_Data.mat'
    data = loadmat(path)
    data_array = data['testdata']

    # transform the testdata to have shape (100, 820)
    data_transposed = data_array.transpose(2, 0, 1)
    data_reshaped = data_transposed.reshape(data_transposed.shape[0], -1)
    test_data = data_reshaped
    return test_data

def center_and_normalize(data):

    """ 
    Centers and normalizes data

    Parameters:
    ----------
    data: A 390 x 100 matrix

    Returns:
    A centered and normalized version of (data).

    """
      # Get dimensions
    M, N = data.shape
    
    # Center the data
    MeanofData = np.mean(data, axis=0, keepdims=True)
    Data_centered = data - MeanofData 
    
    # Normalize the data by dividing by sqrt(M-1)
    Y = Data_centered / np.sqrt(M-1)

    return Y

########################################################################

############################## Synergy Extraction Methods (PCA and RPCA) ##############################
def standard_pca(data):
    '''
    Performs PCA on a 390x100 matrix of data

    This method first caluculates and subtracts 
    the mean across rows to center to center the data.
    Then normalizes the data. Following it performs 
    SVD on the centered and normalized data to extract
    the principal components.

    Parameters: 
    ----------
    data: a 2d numpy array 

    Returns:
    U_T_truncated: principal component matrix -> first k rows are the first k pc's
    num_synergies: k number of principal components required to account >= 95% of the variance 
    _
    _
    _
    
    '''
    
    # Get dimensions
    M, N = data.shape
    
    # Center the data
    MeanofData = np.mean(data, axis=0, keepdims=True)
    Data_centered = data - MeanofData 
    
    # Normalize the data 
    Y = Data_centered / np.sqrt(M-1)
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    
    # Find Number of synergies s.t. the FOV >= .95 #
    fov = np.cumsum(S**2) / np.sum(S**2)
    # print("PCA FOV'S \n")
    # print(fov)
    
    
    num_synergies = 0
    for i in range(len(fov)):
        if fov[i] >= .95:
            num_synergies = i+1
            break
    
    # Vt_truncated = Vt[:num_synergies, :]
    U_T_truncated = U.T[:num_synergies, :]

    return U_T_truncated, num_synergies, U, S, Vt

def get_low_rank(M):
     # Is L_RPCA Actually Low-Rank?
    rank= np.linalg.matrix_rank(M)
    # print(f"Rank: {rank}")

    U, S, Vt = np.linalg.svd(M, full_matrices = True)
    fraction_of_variances = np.cumsum(S**2) / np.sum(S**2)
    # print("RPCA FOV'S \n")
    # print(fraction_of_variances)


    for i in range(fraction_of_variances.shape[0]):
        if fraction_of_variances[i]>=.95:
            # print(f"Num of PC's to account for 95% of the total variance: {i+1}")
            break

    k = i + 1
    # num_synergies = k
    
    # Truncate components
    U_k = U[:, :k]
    # print(f"U_k: {U_k.shape}, U: {U.shape}")
    S_k = np.diag(S[:k])
    # print(f"S_k: {S_k.shape}, S: {S.shape}")
    Vt_k = Vt[:k, :]
    # print(f"Vt_k: {Vt_k.shape}, Vt: {Vt.shape}")
    
    # Construct the Low-Rank Approximation
    L_rpca_reconstructed = np.dot(U_k, np.dot(S_k, Vt_k))

    return L_rpca_reconstructed, U_k.T, k

#######################################################################################################

############################## Simulating Outliers (Rapid Outliers and Occlusions) ##############################
def generate_outliers(
    data: np.ndarray,
    outlier_fraction: float,
    low_mag: float,
    high_mag: float,
    mode: str = "relative",
    seed=None
) -> np.ndarray:
    """
    Generate a matrix of additive outliers.

    Parameters
    ----------
    data : np.ndarray
        Original data array of shape (m, n).
    outlier_fraction : float
        Fraction of total entries to spike (0 <= outlier_fraction <= 1).
    low_mag, high_mag : float
        Bounds for sampling the spike magnitude from Uniform(low_mag, high_mag).
    mode : {'relative', 'absolute'}
        'relative' — spike = ±(mag * original_value)
        'absolute' — spike = ±mag
    random_state : None, int, or np.random.Generator
        Seed or Generator for reproducibility.

    Returns
    -------
    outliers : np.ndarray
        Array of same shape as `data`, with nonzero entries at outlier positions.
    """
    # data = data.T

    
    # Set up RNG
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    m, n = data.shape
    total = m * n

    # Determine number of outliers (at least 1 if fraction > 0)
    k = int(np.round(outlier_fraction * total))
    if outlier_fraction > 0 and k == 0:
        k = 1

    # Pick k unique flat indices
    flat_idxs = rng.choice(total, size=k, replace=False)
    row_idxs, col_idxs = np.unravel_index(flat_idxs, (m, n))

    # Build the outlier matrix
    outliers = np.zeros((m, n), dtype=float)
    for i, j in zip(row_idxs, col_idxs):
        mag = rng.uniform(low_mag, high_mag)
        sign = rng.choice([-1, 1])

        if mode == "relative":
            # scale by the original data value
            outliers[i, j] = sign * mag * data[i, j]
        elif mode == "absolute":
            # constant-magnitude spike
            outliers[i, j] = sign * mag
        else:
            raise ValueError("mode must be 'relative' or 'absolute'")

    return outliers

def simulate_occlusions(V, occlusion_frac=.10, occlusion_type='mild', seed=None):
    """
    Simulate structured occlusion in a grasping dataset.

    Parameters:
        V              : ndarray of shape (n_trials, n_samples), e.g., (100, 390)
        rho_percent    : percentage of samples to occlude per row (0–100)
        occlusion_type : 'mild', 'joint', or 'severe'

    Returns:
        V_occluded     : ndarray with structured occlusions applied
    """
    if seed is not None:
        np.random.seed(seed)
        
    n_trials, n_samples = V.shape
    V_occluded = V.copy()
    
    for i in range(n_trials):
        trial = V[i].copy()
        target = int(occlusion_frac * n_samples)
        occluded_indices = set()

        while len(occluded_indices) < target:
            if occlusion_type == 'mild':
                block_size = np.random.randint(5, 11)     # 5–10 samples
            elif occlusion_type == 'joint':
                block_size = np.random.randint(8, 16)     # 8–15 samples
                # simulate joint: restrict to 1 of 10 equal-length joint segments
                joint_idx = np.random.randint(0, 10)
                joint_start = joint_idx * (n_samples // 10)
                start = np.random.randint(joint_start, joint_start + (n_samples // 10 - block_size + 1))
            elif occlusion_type == 'severe':
                block_size = np.random.randint(20, 41)    # 20–40 samples
                start = np.random.randint(0, n_samples - block_size + 1)
            else:
                raise ValueError("Invalid occlusion type: must be 'mild', 'joint', or 'severe'")

            if occlusion_type != 'joint':
                start = np.random.randint(0, n_samples - block_size + 1)

            block = list(range(start, start + block_size))
            remaining = target - len(occluded_indices)
            if len(block) > remaining:
                block = block[:remaining]
            
            occluded_indices.update(block)

        # Set occluded samples to 0 (or np.nan, depending on your downstream model)
        trial[list(occluded_indices)] = 0.0
        V_occluded[i] = trial

    return V_occluded

################################################################################################################

############################## Reconstruction Methods (LASSO and Robust LASSO) ##############################

def extract_synergies(B_shifted, num_synergies, max_shift):
    # calculate the number of rows per synergy
    rows_per_synergy = max_shift+1

    # calculate the totol rows to extract for the selected synergies
    total_rows = num_synergies * rows_per_synergy

    # sllice the correspidig rows adn return
    return np.array(B_shifted[:total_rows])

def shift_synergies(synergy_matrix):
    T=82
    ts = 39
    n_joints = 10
    max_shift = T-ts
    B_shifted = []
    for synergy in synergy_matrix:
        new_syn_matrix = synergy.reshape(n_joints, ts)
        for shift in range(max_shift+1):
            padded_synergy_matrix = np.zeros((n_joints, T))
            padded_synergy_matrix[:, shift:shift+ts] = new_syn_matrix
            padded_synergy_vector = padded_synergy_matrix.flatten()
            B_shifted.append(padded_synergy_vector)
    return np.array(B_shifted)

def run_lasso_ls_model(B_shifted, V, num_synergies, V_clean):
    count = num_synergies
    max_shift = 43
    while count <= num_synergies:
        B_reduced = extract_synergies(B_shifted, count, max_shift)
        B = B_reduced.T
      # Compute Lasso and LS results
        C_lasso, avg_error_lasso, avg_error_lasso_2, nrmse = compute_coefficients_and_errors_lasso_ls(B, V, V_clean, method="lasso")
        
        # Count non-zero elements
        non_zero_counts_lasso = np.count_nonzero(C_lasso, axis=1)
        # print(f"Lasso nonzero counts: {non_zero_counts_lasso}")
        # non_zero_counts_ls = np.count_nonzero(C_ls, axis=1)      
        
        print(f"Num Synergies: {count}")
        print(f"Mean Synergy Recruitment (Lasso): {np.mean(non_zero_counts_lasso)}")
        
        count+=1

    return avg_error_lasso_2, C_lasso, nrmse


def compute_coefficients_and_errors_lasso_ls(B, V, V_clean, method='lasso', alpha = .00005):
    coefficients = []
    errors = []
    errors_2 = []
    nrmserrors = []
    
    for i in range(V.shape[0]):
        v_i = V[i, :]
        v_i_clean = V_clean[i, :]

        if method == 'lasso':
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter = 10000)
            model.fit(B, v_i)
            c_i = model.coef_
        elif method == 'ls':
            model = LinearRegression(fit_intercept=False)
            model.fit(B, v_i)
            c_i = model.coef_
        v_i_reconstructed = B @ c_i
        error = np.linalg.norm(v_i-v_i_reconstructed)
        error_2 = np.linalg.norm(v_i_clean - v_i_reconstructed)

        nrmse = np.sqrt(mean_squared_error(v_i_clean, v_i_reconstructed)) / ((np.max(v_i_clean) - np.min(v_i_clean)) + 1e-12)
        
        nrmserrors.append(nrmse)
        coefficients.append(c_i)
        errors.append(error)
        errors_2.append(error_2)

        
    return np.array(coefficients), np.mean(errors), np.mean(errors_2), np.mean(nrmserrors)


def run_new_model(B_shifted, test_data_corr, num_synergies, test_data_clean):
    max_shift = 43
    B_reduced = extract_synergies(B_shifted, num_synergies, max_shift)
    B = B_reduced.T
    c_est, e_est, two_norm_errors, nrmse = robust_lasso_cvx(B, test_data_corr, V_CLEAN=test_data_clean)

    non_zero_counts_c = np.count_nonzero(c_est, axis=1)
    non_zero_counts_e = np.count_nonzero(e_est, axis=1)

    print("Anomalies Non-Zero Counts:", np.sum(non_zero_counts_e))
    print(f"Mean Synergy Recruitment (C): {np.mean(non_zero_counts_c)}")

    return np.mean(two_norm_errors), c_est, e_est, np.mean(nrmse)

def robust_lasso_cvx(B, V, lambda_c=0.03, lambda_e=0.05, solver='SCS', V_CLEAN=None):
    coefficients, anomalies, errors, nrmserrors = [], [], [], []

    for i in range(V.shape[0]):
        v_i = V[i, :]
        v_i_clean = V_CLEAN[i, :]
        n, d_c = B.shape

        c = cp.Variable(d_c)
        e = cp.Variable(n)
        residual = v_i - B @ c - e

        objective = cp.Minimize(0.5 * cp.sum_squares(residual) + lambda_c * cp.norm1(c) + lambda_e * cp.norm1(e))
        cp.Problem(objective).solve(solver=solver)

        c_sparse = c.value.copy()
        e_sparse = e.value.copy()
        c_sparse[np.abs(c_sparse) < 1e-4] = 0.0
        e_sparse[np.abs(e_sparse) < 1e-4] = 0.0

        v_i_reconstructed = B @ c_sparse
        approx_error = np.linalg.norm(v_i_clean - v_i_reconstructed)
        nrmse = np.sqrt(mean_squared_error(v_i_clean, v_i_reconstructed)) / ((np.max(v_i_clean) - np.min(v_i_clean))+1e-12)

        coefficients.append(c_sparse)
        anomalies.append(e_sparse)
        errors.append(approx_error)
        nrmserrors.append(nrmse)

    return np.array(coefficients), np.array(anomalies), np.array(errors), np.array(nrmserrors)

#################################################################################### 

############################## Plotting/Visuals ##############################
import numpy as np
import matplotlib.pyplot as plt

def plot_tasks(
    B: np.ndarray,
    C_standard: np.ndarray,
    C_robust: np.ndarray,
    test_data: np.ndarray,
    n_tasks: int = 5,
):
    """
    Show, for the first *n_tasks* rows, the ground truth and two reconstructions:

        • ground truth          = test_data[i]
        • standard LASSO recon  = B @ C_standard[:, i]
        • robust  LASSO recon   = B @ C_robust[:,  i]

    Parameters
    ----------
    B : (d, k) ndarray
        Basis (synergy / loading) matrix.
    C_standard, C_robust : (k, T) ndarray
        Coefficient matrices returned by the two LASSO variants.
        Each column `[:, i]` contains the coefficients for task *i*.
    test_data : (T, n_time) or (n_tasks, n_time) ndarray
        Ground-truth signals—one row per task.
    n_tasks : int, optional
        How many tasks (rows) to plot, starting from 0.  Default is 5.
    """
    n_tasks = min(n_tasks, test_data.shape[0])

    time = np.arange(test_data.shape[1])

    for i in range(n_tasks):
        plt.figure(figsize=(12, 3))

        # --- reconstructions ---
        recon_std   = C_standard[i, :] @ B
        recon_rob   = C_robust[i, :] @ B
        truth       = test_data[i]

        # --- plot ---
        plt.plot(time, recon_std,   label='Standard LASSO recon')
        plt.plot(time, recon_rob,   label='Robust LASSO recon')
        plt.plot(time, truth,       label='Ground truth', linewidth=2, alpha=0.7)

        plt.title(f'Task {i+1}')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

import math

def plot_synergies(M, num_synergies):
    # Number of tasks (should be 100) and number of joints (10)
    num_tasks = 100  # Rows of Vt (100 tasks)
    num_joints = 10
    samples_per_task = 39
    print("Samples per task:", samples_per_task)
    
    # Reshape the data: assuming M has shape (num_synergies * 10 * samples_per_task,)
    # or M is already a matrix with dimensions (num_synergies, 10*samples_per_task)
    # Here, we reshape it to (num_synergies, 10, samples_per_task)
    U9_T_reshaped = M.reshape(num_synergies, num_joints, samples_per_task)
    
    # Dynamically compute grid dimensions for subplots
    ncols = math.ceil(math.sqrt(num_synergies))
    nrows = math.ceil(num_synergies / ncols)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 5))
    axes = np.array(axes).flatten()  # Flatten the axes array for easy iteration
    
    for synergy_idx in range(num_synergies):
        ax = axes[synergy_idx]
        ax.set_title(f"Synergy {synergy_idx + 1}", fontsize=12)
    
        # Plot each joint's profile for this synergy
        for joint_idx in range(num_joints):  # Loop through all 10 joints
            joint_profile = U9_T_reshaped[synergy_idx, joint_idx, :]  # Select profile for the joint
            ax.plot(joint_profile)
    
        ax.grid(True)
    
        # Set x-ticks to show every 10 samples (0, 10, 20, 30)
        ax.set_xticks(np.arange(0, samples_per_task, 10))
        ax.set_xticklabels(np.arange(0, samples_per_task, 10), fontsize=8)
    
    # Hide any unused subplots if the grid is larger than num_synergies
    for i in range(num_synergies, len(axes)):
        axes[i].axis('off')
    
    # plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.show()

def plot_synergies_individually(M, num_synergies):
    """
    Plot each synergy in a separate figure with clearer, thicker lines and darker text.
    
    Each synergy is a row vector of shape (390,) → reshaped to (10, 39) = 10 joints × 39 timepoints.
    
    Args:
        M (np.ndarray): Synergy matrix of shape (num_synergies, 390)
        num_synergies (int): Number of synergies to plot
    """
    num_joints = 10
    samples_per_joint = 39

    assert M.shape == (num_synergies, num_joints * samples_per_joint), \
        f"Expected M.shape = ({num_synergies}, {num_joints * samples_per_joint}), got {M.shape}"

    colors = plt.cm.tab10.colors  # 10 distinct, high-contrast colors

    for idx in range(num_synergies):
        synergy = M[idx].reshape(num_joints, samples_per_joint)

        plt.figure(figsize=(10, 4))
        for joint_idx in range(num_joints):
            plt.plot(
                synergy[joint_idx],
                label=f'Joint {joint_idx+1}',
                linewidth=2,
                color=colors[joint_idx % 10]
            )

        plt.title(
            f'Synergy {idx + 1}',
            fontsize=24,
            color='black'
        )
        plt.xlabel(
            'Time (samples)',
            fontsize=24,
            color='black'
        )
        plt.ylabel(
            'Activation',
            fontsize=24,
            color='black'
        )
        plt.xticks(np.arange(0, samples_per_joint, 10), fontsize=14, color='black')
        plt.yticks(fontsize=14, color='black')
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.tight_layout()
        plt.show()

def shrink(M, tau):
    sgn = np.sign(M)
    S = np.abs(M) - tau
    S[S < 0.0] = 0.0
    return sgn * S

def _svd(method, X, rank, tol, **args):
    rank = min(rank, min(X.shape))
    if method == "approximate":
        # Requires fbpca
        import fbpca
        U, s, Vt = fbpca.pca(X, k=rank, raw=True, **args)
        return U, s, Vt
    elif method == "exact":
        return np.linalg.svd(X, full_matrices=False, **args)
    elif method == "sparse":
        if rank >= min(X.shape):
            return np.linalg.svd(X, full_matrices=False)
        u, s, v = spla.svds(X, k=rank, tol=tol)
        # svds returns ascending order; reverse to descending
        idx = np.argsort(-s)
        s = s[idx]
        u = u[:, idx]
        v = v[idx, :]
        return u, s, v
    else:
        raise ValueError("invalid SVD method")

def pcp(M, delta=1e-6, mu=None, maxiter=500, verbose=True, missing_data=False,
        svd_method="exact", **svd_args):
    """
    Principal Component Pursuit (PCP) via inexact ALM.

    Returns: L, S, (u, s, v) from last SVD.
    """
    # 1) Missing‐data handling
    shape = M.shape
    if missing_data:
        missing = ~(np.isfinite(M))
        M = M.copy()
        M[missing] = 0.0
    else:
        missing = np.zeros_like(M, dtype=bool)
        if not np.all(np.isfinite(M)):
            print("Warning: non‐finite entries present; SVD may fail.")

    m, n = shape

    # 2) Tuning parameters
    lam = 1.0 / np.sqrt(max(m, n))
    if mu is None:
        mu = 0.25 * m * n / np.sum(np.abs(M))
    normM = np.sum(M**2)

    # Initialize
    i = 0
    rank_est = min(m, n)
    S = np.zeros_like(M)
    Y = np.zeros_like(M)

    while True:
        # 3a) SVD step on (M − S + Y/μ)
        T = M - S + (Y / mu)
        u, s, v = _svd(svd_method, T, rank_est + 1, 1.0/mu, **svd_args)

        # 3b) Soft‐threshold singular values
        s_thresh = shrink(s, 1.0 / mu)
        rank_est = np.sum(s_thresh > 0)
        if rank_est == 0:
            L = np.zeros_like(M)
        else:
            u_r = u[:, :rank_est]
            s_r = s_thresh[:rank_est]
            v_r = v[:rank_est, :]
            L = u_r @ np.diag(s_r) @ v_r

        # 3c) Shrinkage for S
        S = shrink(M - L + (Y / mu), lam / mu)

        # 3d) Dual update
        step = M - L - S
        step[missing] = 0.0
        Y += mu * step

        # 3e) Check convergence
        err = np.sqrt(np.sum(step**2) / normM)
        if verbose:
            nnz_S = np.count_nonzero(S)
            # print(f"Iter {i:3d} | err = {err:.3e} | rank = {rank_est:3d} | nnz(S) = {nnz_S:5d}")
        if err < delta or i >= maxiter:
            break
        i += 1

   #  if i >= maxiter and verbose:
        
        # print("Warning: PCP did not converge within maxiter.")

    return L, S, (u, s, v)
  
