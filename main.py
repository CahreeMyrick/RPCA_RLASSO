# Script with Standard Approach and LOSO Appraoch

import pdb
import os
import pathlib
from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.io import loadmat
from numpy.linalg import norm

from Methods import (
    standard_pca,
    generate_outliers,
    simulate_occlusions,
    plot_synergies,          
    get_low_rank,
    run_lasso_ls_model,
    run_new_model,
    shift_synergies,
    pcp,
    center_and_normalize,
    extract_synergies,       
)

# ----------------------------
# Config
# ----------------------------
SUBJECTS: List[int] = [1, 2, 4, 5, 6, 7, 8, 9, 10] # subject 3 not included, there was a problem loading data
OUTLIER_FRACTIONS: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20]
OCCLUSION_TYPES = ['mild', 'joint', 'severe']

base_path      = '/Users/cahree/desktop/Regression/xxxxxxxxxxxxxx/Rapid Natural and ASL tasks/'
test_root      = '/Users/cahree/desktop/Regression/xxxxxxxxxxxxxx/Rapid Natural and ASL tasks/'
output_folder  = '/Users/cahree/desktop/RPCARLASSO_RESULTS'
pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

TOP_K_COMPARE = 3  # compare PC1..PCk between PCA and RPCA

# ----------------------------
# Utilities: paths, save/load
# ----------------------------
def make_synergy_dir(root: str, method: str, outlier_type: str, occ_type: Optional[str]) -> str:
    """
    Creates a directory like:
      RESULTS/SAVED_SYNERGIES/{method}/{outlier_type}/{occ_type_if_any}
    """
    if outlier_type == "Occlusion":
        p = pathlib.Path(root) / "SAVED_SYNERGIES" / method / outlier_type / str(occ_type)
    else:
        p = pathlib.Path(root) / "SAVED_SYNERGIES" / method / outlier_type
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def synergy_filename(dirpath: str, subject: int, frac: float) -> str:
    return os.path.join(dirpath, f"subj_{subject:02d}_frac_{frac:.2f}.npy")

def save_synergies(S: np.ndarray, dirpath: str, subject: int, frac: float) -> None:
    """
    Save synergy matrix S with shape (m, D), where rows are components.
    """
    np.save(synergy_filename(dirpath, subject, frac), S)

def load_synergies(dirpath: str, subject: int, frac: float) -> np.ndarray:
    return np.load(synergy_filename(dirpath, subject, frac))

# ----------------------------
# Similarity metrics
# ----------------------------
def _unit_rows(S: np.ndarray) -> np.ndarray:
    """
    Row-wise unit normalization
    
    - Used as a preproccessing step before doing cosine similarity
    
    Ex: if row is [3, 4], norm = 5 -> normalized row = [.6, .8]
    """
    S = S.copy()
    
    # computes the Eucliden norm (magnitude) of each row
    r = norm(S, axis=1, keepdims=True)

    # prevents division by zero
    r[r == 0] = 1.0

    # devides each row by its corresponding row norm
    # this makes each nonzero row have a unit length 1
    return S / r

def compute_similarity(
    S_pca: np.ndarray,
    S_rpca: np.ndarray,
    k_compare: int = TOP_K_COMPARE,
    flip_polarity: bool = True,
) -> Tuple[list, list]:
    """
    Compare first k matched synergies row-wise (PC1->RPCA1, PC2->RPCA2, ...).

    Returns:
      directional_alignment (list of float)  : cosine similarity
      vector_difference   (list of float)    : L2 norm between unit vectors 
    """
    if S_pca.ndim != 2 or S_rpca.ndim != 2:
        raise ValueError("S_pca and S_rpca must be 2D (components x features).")
    k = min(k_compare, S_pca.shape[0], S_rpca.shape[0])
    if k == 0:
        return [], []

    # unit normalization
    A = _unit_rows(S_pca[:k, :])
    B = _unit_rows(S_rpca[:k, :])

    # comuting cosine similarity (dot product)
    cosines = np.sum(A * B, axis=1).tolist()
    
    diffs = norm(A - B, axis=1).tolist()
    return cosines, diffs

def pad_k(x: list, k: int = TOP_K_COMPARE) -> list:
    return (x + [np.nan] * (k - len(x)))[:k]

# ----------------------------
# Core experiment
# ----------------------------
def experiments(subjects: List[int] = SUBJECTS,
                outlier_fractions: List[float] = OUTLIER_FRACTIONS,
                outlier_type: str = "Rapid",
                occ_type: str = "mild") -> None:
    """
    Runs either Rapid outlier or Occlusion experiments across subjects and fractions.
    Saves:
      - Per-subject metrics CSV (append across fractions)
      - Per-fraction summary CSV
      - Synergy matrices (PCA/RPCA) per subject x fraction x type
    """
    subject_files = [f'subj{subject_num}/ang_vel_mat.csv' for subject_num in subjects]

    # Collect detail rows (per subject & fraction)
    detail_rows = []
    # Collect summary rows (per fraction)
    summary_rows = []

    # Prepare per-subject metrics filename
    if outlier_type == "Occlusion":
        detail_name = f"{outlier_type}_{occ_type}_per_subject_metrics.csv"
        summary_name = f"{outlier_type}_{occ_type}_results_summary.csv"
    else:
        detail_name = f"{outlier_type}_per_subject_metrics.csv"
        summary_name = f"{outlier_type}_results_summary.csv"

    detail_path = os.path.join(output_folder, detail_name)
    summary_path = os.path.join(output_folder, summary_name)

    for frac in outlier_fractions:
        robust_errors, standard_errors = [], []
        num_syns_robust, num_syns_standard = [], []
        nrmse_r, nrmse_l = [], []

        for file_name in subject_files:
            subj_folder = file_name.split('/')[0]     # e.g., 'subj5'
            subject_id = int(subj_folder.replace('subj', ''))

            print(f"\n=== Running Subject: {file_name} | {outlier_type} Corruption = {frac:.2%} ===")

            # ----------------------------
            # Load / reshape test data (Natural)
            # ----------------------------
            mat_path = os.path.join(test_root, subj_folder, 'Natural_Test_Data.mat')
            mat_contents = loadmat(mat_path)
            raw_testdata = mat_contents['testdata']  # shape (10, 82, 100)
            # -> (100 tasks, 10 joints, 82 time) → flatten to (100, 820)
            test_data = raw_testdata.transpose(2, 0, 1).reshape(100, -1)

            # ----------------------------
            # Load training (Rapid) data for THIS subject
            # ----------------------------
            training_data_path = os.path.join(base_path, file_name)
            training_data = pd.read_csv(training_data_path).values  # shape (100, 390) typically

            # ----------------------------
            # Inject corruption (Rapid vs Occlusion)
            # ----------------------------
            rand_seed = np.random.randint(0, 1e6)
            if outlier_type == "Rapid":
                outlier_matrix_train = generate_outliers(training_data, frac, low_mag=2, high_mag=4, mode='relative', seed=rand_seed)
                training_data_corrupt = training_data + outlier_matrix_train

                outlier_matrix_test = generate_outliers(test_data, frac, low_mag=2, high_mag=4, mode='relative', seed=rand_seed + 1)
                testing_data_corrupt = test_data + outlier_matrix_test
            else:
                training_data_corrupt = simulate_occlusions(training_data, frac, occ_type, seed=rand_seed)
                testing_data_corrupt  = simulate_occlusions(test_data, frac, occ_type, seed=rand_seed + 1)

            # ----------------------------
            # STANDARD PCA synergies
            # ----------------------------
            #standard_pca expects features x samples 
            synergy_matrix_pca, num_synergies_pca, U_pca, _, _ = standard_pca(training_data_corrupt.T)
            
            # synergies as rows
            S_pca = U_pca.T[:num_synergies_pca]  # (m_pca, D)

            # ----------------------------
            # ROBUST PCA synergies via PCP
            # ----------------------------
            Y = center_and_normalize(training_data_corrupt.T)
            L_rpca, _, _ = pcp(Y)
            _, synergy_matrix_rpca, num_synergies_rpca = get_low_rank(L_rpca)
            S_rpca = synergy_matrix_rpca  # (m_rpca, D), rows are synergies

            # ----------------------------
            # Save synergies (before time-shift)
            # ----------------------------
            pca_dir  = make_synergy_dir(output_folder, "PCA",  outlier_type, occ_type if outlier_type == "Occlusion" else None)
            rpca_dir = make_synergy_dir(output_folder, "RPCA", outlier_type, occ_type if outlier_type == "Occlusion" else None)
            save_synergies(S_pca,  pca_dir,  subject_id, frac)
            save_synergies(S_rpca, rpca_dir, subject_id, frac)

            # ----------------------------
            # Build time-shifted dictionaries
            # ----------------------------
            # For fairness, use the same m when running LASSO vs RLASSO (you did this already)
            B_pca  = shift_synergies(S_pca[:num_synergies_rpca])
            B_rpca = shift_synergies(S_rpca)

            # ----------------------------
            # Modeling
            # ----------------------------
            avg_error_l, C_l, avg_nrmse_l = run_lasso_ls_model(B_pca, testing_data_corrupt, num_synergies_rpca, test_data)
            avg_error_r, C_r, _, avg_nrmse_r = run_new_model(B_rpca, testing_data_corrupt, num_synergies_rpca, test_data)

            # displaying results
            print(f"Standard LASSO: Avg 2-norm Error: {avg_error_l:.4f} | Avg nrmse: {avg_nrmse_l: .4f}")
            print(f"Robust LASSO mean-test error: {avg_error_r:.4f}| Avg nrmse: {avg_nrmse_r: .4f}")

            # ----------------------------
            # Similarity metrics (top-k)
            # ----------------------------
            dir_align, vec_diff = compute_similarity(S_pca, S_rpca, k_compare=TOP_K_COMPARE)

            # ----------------------------
            # Accumulate per-subject detail
            # ----------------------------
            detail_rows.append({
                'subject': subject_id,
                'corruption': frac,
                'outlier_type': outlier_type,
                'occlusion_type': occ_type if outlier_type == "Occlusion" else None,
                'lasso_mean_subject': avg_error_l,
                'robust_lasso_mean_subject': avg_error_r,
                'num_synergies_pca': num_synergies_pca,
                'num_synergies_rpca': num_synergies_rpca,
                'dir_PC1': dir_align[0], 'dir_PC2': dir_align[1], 'dir_PC3': dir_align[2],
                'diff_PC1': vec_diff[0],  'diff_PC2': vec_diff[1],  'diff_PC3': vec_diff[2],
            })

            # ----------------------------
            # For per-fraction summary
            # ----------------------------
            standard_errors.append(avg_error_l)
            robust_errors.append(avg_error_r)
            num_syns_standard.append(num_synergies_pca)
            num_syns_robust.append(num_synergies_rpca)
            nrmse_r.append(avg_nrmse_r)
            nrmse_l.append(avg_nrmse_l)

        # accumulating results across subjects
        mean_2norm_robust = np.mean(robust_errors)
        std_2norm_robust = np.std(robust_errors)
        mean_2norm_standard = np.mean(standard_errors)
        std_2norm_standard = np.std(standard_errors)
        mean_nrmse_robust = np.mean(nrmse_r)
        std_nrmse_robust = np.std(nrmse_r)
        mean_nrmse_standard = np.mean(nrmse_l)
        std_nrmse_standard = np.std(nrmse_l)
        mean_num_robust_syns = np.mean(num_syns_robust)
        mean_num_standard_syns = np.mean(num_syns_standard)

        row = {
            'corruption': frac,
            'lasso_2norm_mean': mean_2norm_standard,
            'lasso_2nrom_std': std_2norm_standard,
            'robust_lasso_2norm_mean': mean_2norm_robust,
            'robust_lasso_2norm_std': std_2norm_robust,
            'lasso_mean_nrmse': mean_nrmse_standard,
            'lasso_std_nrmse': std_nrmse_standard,
            'robust_lasso_mean_nrmse': mean_nrmse_robust,
            'robust_lasso_std_nrmse': std_nrmse_robust,
            'num_synergies_robust_mean': mean_num_robust_syns,
            'num_synergies_standard_mean': mean_num_standard_syns,
        }
        if outlier_type == "Occlusion":
            row['occlusion_type'] = occ_type
        summary_rows.append(row)

        # ----------------------------
        # Write/append per-subject detail for this fraction
        # ----------------------------
        this_frac_detail = [r for r in detail_rows
                            if r['corruption'] == frac and
                               r['outlier_type'] == outlier_type and
                               (r['occlusion_type'] == (occ_type if outlier_type == "Occlusion" else None))]

        detail_df = pd.DataFrame(this_frac_detail)
        if os.path.exists(detail_path):
            # append rows without header
            detail_df.to_csv(detail_path, mode='a', header=False, index=False)
        else:
            detail_df.to_csv(detail_path, index=False)
        print(f"Per-subject metrics appended to {os.path.basename(detail_path)}")

        # ----------------------------
        # Write per-fraction summary so far 
        # ----------------------------
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to {os.path.basename(summary_path)}")

    # ----------------------------
    # Also write an averaged similarity table (mean across subjects per fraction)
    # ----------------------------
    full_detail_df = pd.read_csv(detail_path)
    group_cols = ['corruption'] + (['occlusion_type'] if outlier_type == "Occlusion" else [])
    sim_agg = full_detail_df.groupby(group_cols).agg({
        'dir_PC1': 'mean', 'dir_PC2': 'mean', 'dir_PC3': 'mean',
        'diff_PC1': 'mean', 'diff_PC2': 'mean', 'diff_PC3': 'mean'
    }).reset_index()

    sim_name = (f"{outlier_type}_{occ_type}_similarity_averaged.csv"
                if outlier_type == "Occlusion"
                else f"{outlier_type}_similarity_averaged.csv")
    sim_path = os.path.join(output_folder, sim_name)
    sim_agg.to_csv(sim_path, index=False)
    print(f"Averaged similarity saved to {os.path.basename(sim_path)}")

# ----------------------------
# LOSO-CV (Leave-One-Subject-Out)
# ----------------------------
def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    if os.path.getsize(path) == 0:
        raise EmptyDataError(f"CSV is empty: {path}")
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        # In case of whitespace-only or corrupted header
        raise EmptyDataError(f"No columns to parse in CSV: {path}")

def _load_subject_train_test(subject_id: int) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load Rapid (training) CSV and Natural (test) .mat for a subject.
    Returns:
      training_data: 2D array (num_tasks, D_train)
      test_data:     2D array (num_tasks, 820)  from Natural_Test_Data.mat as in your script
      subj_folder:   'subjX'
    """
    subj_folder = f"subj{subject_id}"
    
    # Training CSV
    training_csv = os.path.join(base_path, subj_folder, "ang_vel_mat.csv")
    
    # print(f"[DEBUG] Loading training CSV for subject {subject_id}: {training_csv}")
    training_df = _safe_read_csv(training_csv)
    training_data = training_df.values
    
    # Test (Natural) MAT
    mat_path = os.path.join(test_root, subj_folder, "Natural_Test_Data.mat")
    mat_contents = loadmat(mat_path)
    raw_testdata = mat_contents["testdata"]        # (10, 82, 100)
    test_data = raw_testdata.transpose(2, 0, 1).reshape(100, -1)  # (100, 820)

    return training_data, test_data, subj_folder


def loso_experiments(subjects: List[int] = SUBJECTS,
                     outlier_fractions: List[float] = OUTLIER_FRACTIONS,
                     outlier_type: str = "Rapid",
                     occ_type: str = "mild") -> None:
    """
    Leave-One-Subject-Out evaluation:
      For each subject s as TEST:
        - Build training pool from all subjects except s.
        - Fit PCA/RPCA synergies on the pooled TRAIN data (with corruption added for this fold).
        - Corrupt TEST subject Natural test data for input; use clean Natural data as ground truth.
        - Evaluate LASSO vs Robust model, accumulate per-fold/per-fraction metrics.

    Output files mirror experiments():
      - Per-fold detail CSV (rows = (test_subject, fraction, metrics))
      - Per-fraction summary CSV (mean±std across held-out subjects)
    """
    # Output names
    if outlier_type == "Occlusion":
        detail_name  = f"LOSO_{outlier_type}_{occ_type}_per_subject_metrics.csv"
        summary_name = f"LOSO_{outlier_type}_{occ_type}_results_summary.csv"
    else:
        detail_name  = f"LOSO_{outlier_type}_per_subject_metrics.csv"
        summary_name = f"LOSO_{outlier_type}_results_summary.csv"
    detail_path  = os.path.join(output_folder, detail_name)
    summary_path = os.path.join(output_folder, summary_name)

    # reset files
    if os.path.exists(detail_path):
        os.remove(detail_path)
    if os.path.exists(summary_path):
        os.remove(summary_path)

    summary_rows_all_fracs = []

    for frac in outlier_fractions:
        print(f"\n================ LOSO @ corruption {frac:.2%} ({outlier_type}{'/' + occ_type if outlier_type=='Occlusion' else ''}) ================\n")

        fold_errors_lasso  = []
        fold_errors_robust = []
        fold_num_syn_pca   = []
        fold_num_syn_rpca  = []
        fold_nrmse_r = []
        fold_nrmse_l = []

        detail_rows_this_frac = []

        for test_subject in subjects:
            # Build train/test splits at the SUBJECT level
            train_subjects = [s for s in subjects if s != test_subject]

            # ---- Load & stack TRAIN for all train_subjects ----
            train_mats = []
            for s in train_subjects:
                trn, _, _ = _load_subject_train_test(s)
                train_mats.append(trn)
            X_train = np.vstack(train_mats)  # (N_train_tasks_total, D_train)

            # ---- Load TEST for the held-out subject ----
            _, X_test_clean, subj_folder = _load_subject_train_test(test_subject)

            # ---- Inject corruption per fold ----
            fold_seed = np.random.randint(0, 1e6)

            if outlier_type == "Rapid":
                # corrupt TRAIN and TEST for input
                X_train_corrupt = generate_outliers(
                    X_train, frac, low_mag=2, high_mag=4, mode='relative', seed=fold_seed
                )
                X_train_corrupt = X_train + X_train_corrupt

                X_test_corrupt  = generate_outliers(
                    X_test_clean, frac, low_mag=2, high_mag=4, mode='relative', seed=fold_seed + 1
                )
                X_test_input = X_test_clean + X_test_corrupt

            else:  # Occlusion
                X_train_corrupt = simulate_occlusions(X_train, frac, occ_type, seed=fold_seed)
                X_test_input    = simulate_occlusions(X_test_clean, frac, occ_type, seed=fold_seed + 1)

            # ---- Fit PCA/RPCA synergies on TRAIN ONLY (transpose = features x samples) ----
            # PCA
            synergy_matrix_pca, num_synergies_pca, U_pca, _, _ = standard_pca(X_train_corrupt.T)
            S_pca = U_pca.T[:num_synergies_pca]  # (m_pca, D)

            # RPCA
            Y = center_and_normalize(X_train_corrupt.T)
            L_rpca, _, _ = pcp(Y)
            _, synergy_matrix_rpca, num_synergies_rpca = get_low_rank(L_rpca)
            S_rpca = synergy_matrix_rpca  # (m_rpca, D)

            # Keep same num synergues for both approaches
            k = int(num_synergies_rpca)

            # Build time-shifted dictionaries
            B_pca  = shift_synergies(S_pca[:k])
            B_rpca = shift_synergies(S_rpca)

            # save synergies per fold/test_subject
            pca_dir  = make_synergy_dir(output_folder, "PCA_LOSO",  outlier_type, occ_type if outlier_type == "Occlusion" else None)
            rpca_dir = make_synergy_dir(output_folder, "RPCA_LOSO", outlier_type, occ_type if outlier_type == "Occlusion" else None)
            save_synergies(S_pca,  pca_dir,  test_subject, frac)
            save_synergies(S_rpca, rpca_dir, test_subject, frac)

            # ---- Evaluate on held-out subject ----
            # (B, X_corrupted_input, k, X_clean_ground_truth)
            avg_error_l, C_l, avg_nrmse_l = run_lasso_ls_model(B_pca,  X_test_input, k, X_test_clean)
            avg_error_r, C_r, _, avg_nrmse_r = run_new_model(B_rpca, X_test_input, k, X_test_clean)

            # displaying results
            print(f"[LOSO] Test subj {test_subject:>2}: ")
            print(f"Standard LASSO: Avg 2-norm Error: {avg_error_l:.4f} | Avg nrmse: {avg_nrmse_l: .4f}")
            print(f"Robust LASSO mean-test error: {avg_error_r:.4f}| Avg nrmse: {avg_nrmse_r: .4f}")

            # Similarity diagnostics (optional): compare top-k PCs from PCA vs RPCA
            dir_align, vec_diff = compute_similarity(S_pca, S_rpca, k_compare=TOP_K_COMPARE)
        
            # Collect per-fold rows
            detail_rows_this_frac.append({
                'test_subject': test_subject,
                'corruption': frac,
                'outlier_type': outlier_type,
                'occlusion_type': occ_type if outlier_type == "Occlusion" else None,
                'lasso_mean_subject': float(avg_error_l),
                'robust_lasso_mean_subject': float(avg_error_r),
                'num_synergies_pca': int(num_synergies_pca),
                'num_synergies_rpca': int(num_synergies_rpca),
                'dir_PC1': dir_align[0], 'dir_PC2': dir_align[1], 'dir_PC3': dir_align[2],
                'diff_PC1': vec_diff[0],  'diff_PC2': vec_diff[1],  'diff_PC3': vec_diff[2],
            })

            fold_errors_lasso.append(float(avg_error_l))
            fold_errors_robust.append(float(avg_error_r))
            fold_num_syn_pca.append(int(num_synergies_pca))
            fold_num_syn_rpca.append(int(num_synergies_rpca))
            fold_nrmse_l.append(float(avg_nrmse_l))
            fold_nrmse_r.append(float(avg_nrmse_r))
            

        # ---- Per-fraction summary across folds (held-out subjects) ----
        mean_2norm_l = float(np.mean(fold_errors_lasso))
        std_2norm_l  = float(np.std(fold_errors_lasso)) if len(fold_errors_lasso) > 1 else 0.0
        mean_2norm_r = float(np.mean(fold_errors_robust))
        std_2norm_r  = float(np.std(fold_errors_robust)) if len(fold_errors_robust) > 1 else 0.0
        mean_nrmse_l = float(np.mean(fold_nrmse_l))
        std_nrmse_l = float(np.std(fold_nrmse_l))
        mean_nrmse_r = float(np.mean(fold_nrmse_r))
        std_nrmse_r = float(np.std(fold_nrmse_r))
        mean_num_robust_syns = float(np.mean(fold_num_syn_rpca))
        mean_num_standard_syns = float(np.mean(fold_num_syn_pca))
        
        

        # print(f"\n[LOSO] Finished fraction {frac:.2%} → LASSO: {mean_l:.4f} ± {std_l:.4f} | Robust: {mean_r:.4f} ± {std_r:.4f}\n")

        # Append per-fold details to CSV
        detail_df = pd.DataFrame(detail_rows_this_frac)
        if os.path.exists(detail_path):
            detail_df.to_csv(detail_path, mode='a', header=False, index=False)
        else:
            detail_df.to_csv(detail_path, index=False)
        print(f"LOSO per-subject metrics appended to {os.path.basename(detail_path)}")

        # Update running summary table
        summary_row = {
            'corruption': frac,
            'occlusion_type': occ_type,
            'lasso_mean_2-norm_error':  mean_2norm_l,
            'lasso_std_2-norm error':   std_2norm_l ,
            'lasso_mean_nrmse': mean_nrmse_l,
            'lasso_std_nrmse': std_nrmse_l,
            'robust_lasso_mean_2-norm_error':mean_2norm_r,
            'robust_lasso_std_2-norm_error': std_2norm_r,
            'robust_lasso_mean_nrmse': mean_nrmse_r,
            'robust_lasso_std_nrmse': std_nrmse_r,
            'num_synergies_robust_mean': mean_num_robust_syns,
            'num_synergies_standard_mean': mean_num_standard_syns,
        }
        if outlier_type == "Occlusion":
            summary_row['occlusion_type'] = occ_type
        summary_rows_all_fracs.append(summary_row)

        # Write summary so far (overwrite for clarity)
        pd.DataFrame(summary_rows_all_fracs).to_csv(summary_path, index=False)
        print(f"LOSO summary saved to {os.path.basename(summary_path)}")

    # aggregated similarity across folds / fractions
    full_detail_df = pd.read_csv(detail_path)
    group_cols = ['corruption'] + (['occlusion_type'] if outlier_type == "Occlusion" else [])
    sim_agg = full_detail_df.groupby(group_cols).agg({
        'dir_PC1':'mean','dir_PC2':'mean','dir_PC3':'mean',
        'diff_PC1':'mean','diff_PC2':'mean','diff_PC3':'mean'
    }).reset_index()
    sim_name = (f"LOSO_{outlier_type}_{occ_type}_similarity_averaged.csv"
                if outlier_type == "Occlusion"
                else f"LOSO_{outlier_type}_similarity_averaged.csv")
    sim_path = os.path.join(output_folder, sim_name)
    sim_agg.to_csv(sim_path, index=False)
    print(f"LOSO averaged similarity saved to {os.path.basename(sim_path)}")

# ----------------------------
# Main
# ----------------------------
def main():
    # Occlusion experiments
    for occ in OCCLUSION_TYPES:
        experiments(outlier_type="Occlusion", occ_type=occ)
        
    # Rapid outlier experiments
    experiments(outlier_type="Rapid")
    
    # LOSO with Rapid outliers
    loso_experiments(outlier_type="Rapid")
    
    # LOSO with Occlusions (run each subtype if desired)
    for occ in OCCLUSION_TYPES:
        loso_experiments(outlier_type="Occlusion", occ_type=occ)

if __name__ == "__main__":
    main()


