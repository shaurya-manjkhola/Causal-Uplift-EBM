# =============================================================================
# config.py
# -----------------------------------------------------------------------------
# Single source of truth for every constant used across the project.
# Import this in EVERY script and notebook so that results are reproducible:
#
#   from src.config import GLOBAL_SEED, DATA_PATH, TARGET_COL, TREATMENT_COL
#
# WHY THIS FILE EXISTS:
#   If each notebook hard-codes random_state=42 independently, one teammate
#   accidentally writes 0 or omits it entirely and the CV scores change every
#   run.  One import fixes that permanently.
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

GLOBAL_SEED = 42
# Pass this into: train_test_split, StratifiedKFold, XGBClassifier,
# ExplainableBoostingClassifier, GridSearchCV, numpy random calls — everywhere.

# ---------------------------------------------------------------------------
# File paths  (relative to repo root — works on any machine after git clone)
# ---------------------------------------------------------------------------

# Directory that contains the raw and processed CSVs
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Raw Hillstrom file — place the downloaded CSV here
RAW_DATA_PATH = os.path.join(DATA_DIR, "hillstrom.csv")

# Processed file written by data_loader.py after encoding
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "hillstrom_processed.csv")

# Where serialised .pkl model files are saved (Member 3 writes, Member 4/5 read)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Where plots and metric CSVs are saved
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

# ---------------------------------------------------------------------------
# Dataset column names  (define once — avoids silent typo bugs)
# ---------------------------------------------------------------------------

# The binary treatment flag: 1 = received email campaign, 0 = control group
TREATMENT_COL = "segment"        # raw column name before we re-map it

# Primary target we model in this project (binary: did user visit site?)
TARGET_COL = "visit"

# Alternative target (conversion is sparser ~1% — useful sensitivity check)
CONVERSION_COL = "conversion"

# Categorical columns that need One-Hot Encoding
CATEGORICAL_COLS = ["zip_code", "channel"]

# Numerical columns — kept raw (EBMs learn their own step functions)
NUMERICAL_COLS = ["recency", "history"]

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

TEST_SIZE   = 0.20   # 80% train, 20% test
# Stratify on a combined treatment+target stratum so all four groups
# (treated-converted, treated-not, control-converted, control-not) are
# proportionally represented in both splits.

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

CV_FOLDS = 5
# StratifiedKFold with shuffle=True and random_state=GLOBAL_SEED is used
# everywhere — never plain KFold, because the positive class rate is ~15%
# for 'visit' and ~1% for 'conversion'.  Plain KFold can produce folds with
# zero positives, making CV scores meaningless.