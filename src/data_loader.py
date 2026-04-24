# =============================================================================
# data_loader.py  —  Member 1: Data Engineer
# -----------------------------------------------------------------------------
# This script does four things:
#   1. Loads the raw Hillstrom CSV and validates its integrity.
#   2. Engineers features  (One-Hot Encoding of categoricals, raw numerics).
#   3. Creates a stratified 80/20 train/test split that respects both the
#      treatment flag AND the target variable simultaneously.
#   4. Runs a quick propensity-score overlap check so we can honestly say
#      the T-Learner's identification assumptions hold on this dataset.
#
# USAGE (from any notebook in /notebooks/):
#   import sys; sys.path.insert(0, '..')
#   from src.data_loader import load_and_preprocess_hillstrom, check_propensity_overlap
#   X_train, X_test, y_train, y_test, t_train, t_test = load_and_preprocess_hillstrom()
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import the single source of truth for all constants.
# The leading dot (.) is a RELATIVE import — it tells Python to look inside
# the src/ package itself rather than searching system libraries.
# This is required because data_loader.py and config.py live in the same folder.
try:
    from .config import (                      # relative import (when used as package)
        GLOBAL_SEED, RAW_DATA_PATH, PROCESSED_DATA_PATH,
        TREATMENT_COL, TARGET_COL,
        CATEGORICAL_COLS, NUMERICAL_COLS,
        TEST_SIZE
    )
except ImportError:
    from config import (                       # fallback for direct script execution
        GLOBAL_SEED, RAW_DATA_PATH, PROCESSED_DATA_PATH,
        TREATMENT_COL, TARGET_COL,
        CATEGORICAL_COLS, NUMERICAL_COLS,
        TEST_SIZE
    )


# =============================================================================
# STEP 1  —  Load & validate
# =============================================================================

def _load_raw(filepath: str) -> pd.DataFrame:
    """
    Reads the CSV, checks shape and missing values, and re-maps the
    Hillstrom 'segment' column into a clean binary treatment flag.

    The original 'segment' column has three values:
        "No E-Mail"           → treatment = 0  (control group)
        "Mens E-Mail"         → treatment = 1  (treated)
        "Womens E-Mail"       → treatment = 1  (treated)
    We merge both email groups into treatment=1 because the project focuses
    on ANY ad exposure vs no exposure.  If you want to split Men vs Women
    later, you can do so using the original 'segment' column which is kept.
    """
    print(f"[1/5] Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    # ── Dimensionality check ─────────────────────────────────────────────────
    print(f"      Shape: {df.shape}")
    assert df.shape[0] == 64_000, (
        f"Expected 64,000 rows, got {df.shape[0]}. "
        "Check you are using the correct Hillstrom dataset."
    )

    # ── Missing value check ──────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"[WARN] Missing values detected:\n{null_counts[null_counts > 0]}")
        # Drop rows with any nulls — document this decision explicitly
        df = df.dropna()
        print(f"      Rows after dropping nulls: {len(df)}")
    else:
        print("      No missing values found. Good.")

    # ── Create clean binary treatment column ─────────────────────────────────
    # 'treatment' = 1 if the user received any email, 0 if control group
    df["treatment"] = (df[TREATMENT_COL] != "No E-Mail").astype(int)

    print(f"      Treatment distribution:\n{df['treatment'].value_counts()}")
    print(f"      Target ('{TARGET_COL}') distribution:\n{df[TARGET_COL].value_counts()}")

    return df


# =============================================================================
# STEP 2  —  Feature engineering
# =============================================================================

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies One-Hot Encoding to categorical columns and keeps numerical
    columns completely untouched.

    WHY NO SCALING ON NUMERICS:
        Explainable Boosting Machines learn pairwise step-functions over
        the raw feature values.  If we scale 'recency', the shape functions
        (f_i) will be expressed in z-score units, which makes the EBM's
        built-in plots uninterpretable to a business audience.
        XGBoost is also scale-invariant (it uses rank-based splits).
        So scaling would only hurt interpretability with zero accuracy gain.

    WHY drop_first=True IN ONE-HOT ENCODING:
        With k categories, k dummies are perfectly multicollinear
        (they sum to 1 always).  Dropping the first prevents the
        'dummy variable trap', which inflates standard errors in linear
        models and can create redundant splits in tree models.
    """
    print("[2/5] Engineering features...")

    # One-Hot Encode categorical columns
    df_encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        drop_first=True,     # avoids multicollinearity — see docstring
        dtype=int            # store as integers (0/1) not booleans
    )

    # Convert 'newbie' to int if it exists and is boolean
    if "newbie" in df_encoded.columns:
        df_encoded["newbie"] = df_encoded["newbie"].astype(int)

    print(f"      Feature matrix columns after encoding: {list(df_encoded.columns)}")
    return df_encoded


# =============================================================================
# STEP 3  —  Stratified train / test split
# =============================================================================

def _stratified_split(df: pd.DataFrame):
    """
    Splits data 80/20 using a combined strata label so both the treatment
    flag AND the target class are proportionally represented in each split.

    WHY COMBINED STRATA:
        We have 4 meaningful groups:
            (treatment=0, visit=0) — control, did not visit
            (treatment=0, visit=1) — control, did visit
            (treatment=1, visit=0) — treated, did not visit
            (treatment=1, visit=1) — treated, did visit  ← rarest group
        Stratifying on visit alone would not guarantee proportional
        treatment representation.  Stratifying on treatment alone would
        not guarantee proportional class balance.
        The combined label fixes both simultaneously.
    """
    print("[3/5] Creating stratified 80/20 train/test split...")

    # Identify feature columns: drop target, treatment raw col, and binary treatment
    drop_cols = [TARGET_COL, "conversion", TREATMENT_COL, "treatment"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].values
    treatment = df["treatment"].values

    # Combined strata: "0_0", "0_1", "1_0", "1_1"
    strata = pd.Series(treatment).astype(str) + "_" + pd.Series(y).astype(str)

    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, treatment,
        test_size=TEST_SIZE,
        random_state=GLOBAL_SEED,
        stratify=strata          # the key line — see docstring
    )

    print(f"      Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"      Train treatment rate: {t_train.mean():.3f}  |  Test: {t_test.mean():.3f}")
    print(f"      Train target rate:    {y_train.mean():.3f}  |  Test: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test, t_train, t_test


# =============================================================================
# STEP 4  —  Propensity score overlap check
# =============================================================================

def check_propensity_overlap(X_train, t_train, save_path: str = None):
    """
    Fits a simple logistic regression that tries to predict treatment
    from features.  In a perfectly randomised experiment (like Hillstrom),
    the propensity scores for treated and control groups should almost
    completely overlap.

    WHY THIS MATTERS FOR THE T-LEARNER:
        The T-Learner assumes that for every feature combination X, both
        treated AND control users exist (the 'overlap' or 'positivity'
        assumption).  If certain users only ever appear in one group,
        the T-Learner extrapolates outside its training data for those
        regions, producing unreliable CATE estimates.

        Showing this plot to the panel proves you understand causal
        identification — most students never think about this.

    EXPECTED RESULT for Hillstrom:
        Since Hillstrom is an RCT (randomised controlled trial), the
        two distributions should look almost identical.  That's the point —
        randomisation guarantees overlap.  The plot is the proof.

    Args:
        X_train:   feature matrix (pandas DataFrame)
        t_train:   binary treatment vector
        save_path: if provided, saves the plot as a PNG to this path
    """
    print("[4/5] Running propensity score overlap check...")

    # Scale features for logistic regression (unlike EBM/XGB, LogReg needs it)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Fit propensity model: predict P(T=1 | X)
    propensity_model = LogisticRegression(
        max_iter=1000,
        random_state=GLOBAL_SEED
    )
    propensity_model.fit(X_scaled, t_train)

    # Propensity score = predicted probability of being treated
    propensity_scores = propensity_model.predict_proba(X_scaled)[:, 1]

    # Separate scores by group
    scores_treated = propensity_scores[t_train == 1]
    scores_control = propensity_scores[t_train == 0]

    # Plot overlapping histograms
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores_treated, bins=40, alpha=0.5, color="#1D9E75", label="Treated (T=1)", density=True)
    ax.hist(scores_control, bins=40, alpha=0.5, color="#378ADD", label="Control (T=0)", density=True)
    ax.set_xlabel("Propensity score  P(T=1 | X)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Propensity score overlap check\n(should overlap substantially for RCT data)", fontsize=12)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"      Propensity plot saved to: {save_path}")

    plt.show()

    # Quantitative summary
    print(f"      Treated  — mean propensity: {scores_treated.mean():.3f}  std: {scores_treated.std():.3f}")
    print(f"      Control  — mean propensity: {scores_control.mean():.3f}  std: {scores_control.std():.3f}")
    print("      If these are very close, the overlap assumption holds. Good.")

    return propensity_scores


# =============================================================================
# STEP 5  —  Save processed data
# =============================================================================

def _save_processed(df: pd.DataFrame):
    """Saves the fully encoded DataFrame to /data/hillstrom_processed.csv"""
    print(f"[5/5] Saving processed data to: {PROCESSED_DATA_PATH}")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("      Done.")


# =============================================================================
# PUBLIC INTERFACE  —  this is what every other script calls
# =============================================================================

def load_and_preprocess_hillstrom(filepath: str = RAW_DATA_PATH, save: bool = True):
    """
    Master function.  Runs all five steps and returns the six arrays
    needed by every downstream member.

    Returns:
        X_train     : pd.DataFrame  — training features
        X_test      : pd.DataFrame  — test features  (NEVER touch this during training)
        y_train     : np.ndarray    — binary target (visit) for training
        y_test      : np.ndarray    — binary target for evaluation
        t_train     : np.ndarray    — binary treatment flag for training
        t_test      : np.ndarray    — binary treatment flag for evaluation

    Example:
        from src.data_loader import load_and_preprocess_hillstrom
        X_train, X_test, y_train, y_test, t_train, t_test = load_and_preprocess_hillstrom()
    """
    df_raw      = _load_raw(filepath)
    df_encoded  = _engineer_features(df_raw)

    if save:
        _save_processed(df_encoded)

    X_train, X_test, y_train, y_test, t_train, t_test = _stratified_split(df_encoded)

    print("\n=== Data pipeline complete ===")
    print(f"  X_train shape : {X_train.shape}")
    print(f"  X_test shape  : {X_test.shape}")
    print(f"  Feature list  : {list(X_train.columns)}\n")

    return X_train, X_test, y_train, y_test, t_train, t_test