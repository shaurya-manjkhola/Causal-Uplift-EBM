# =============================================================================
# integration_test.py  —  Member 2 deliverable
# -----------------------------------------------------------------------------
# Run this from the REPO ROOT before every commit:
#   python integration_test.py
#
# PURPOSE:
#   Verifies that all five members' components plug together correctly
#   using a tiny synthetic dataset (200 rows).  No real data needed.
#   Takes < 30 seconds.  If this passes, the real pipeline will run.
#
#   This is the single highest-ROI addition in the project.
#   Five notebooks written in parallel WILL silently break each other —
#   this catches those breaks before demo day.
# =============================================================================

import sys
import os
import numpy as np
import traceback

# ── Add src/ to path ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

PASS = "  PASS"
FAIL = "  FAIL"

def run_test(name, fn):
    """Runs a test function, prints PASS/FAIL, returns True/False."""
    try:
        fn()
        print(f"{PASS} — {name}")
        return True
    except Exception as e:
        print(f"{FAIL} — {name}")
        print(f"       Error: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# Test data: 200 rows, random binary target, random binary treatment
# =============================================================================

def make_dummy_data(n=200, n_features=8, seed=42):
    """
    Creates a tiny synthetic dataset that mimics the Hillstrom structure.
    Used ONLY for integration testing — never for real model training.
    """
    rng = np.random.default_rng(seed)
    X         = rng.standard_normal((n, n_features)).astype(np.float32)
    y         = rng.integers(0, 2, size=n)          # binary target
    treatment = rng.integers(0, 2, size=n)           # binary treatment
    return X, y, treatment


# =============================================================================
# Test 1: config.py imports correctly
# =============================================================================

def test_config_imports():
    from config import GLOBAL_SEED, TEST_SIZE, CV_FOLDS
    assert isinstance(GLOBAL_SEED, int), "GLOBAL_SEED must be int"
    assert 0 < TEST_SIZE < 1,           "TEST_SIZE must be between 0 and 1"
    assert CV_FOLDS >= 2,               "CV_FOLDS must be at least 2"


# =============================================================================
# Test 2: CustomTLearner accepts sklearn estimators
# =============================================================================

def test_tlearner_with_logreg():
    from custom_tlearner import CustomTLearner
    from sklearn.linear_model import LogisticRegression

    X, y, treatment = make_dummy_data()

    lr = LogisticRegression(max_iter=200, random_state=42)
    tl = CustomTLearner(treatment_model=lr, control_model=lr)

    tl.fit(X, y, treatment)

    uplift = tl.predict_uplift(X)

    assert uplift.shape == (200,),  f"Expected shape (200,), got {uplift.shape}"
    assert uplift.dtype in [np.float32, np.float64], "Uplift must be float"
    assert not np.any(np.isnan(uplift)), "NaN in uplift predictions"


# =============================================================================
# Test 3: Input validation raises correct errors
# =============================================================================

def test_input_validation_length_mismatch():
    from custom_tlearner import CustomTLearner
    from sklearn.linear_model import LogisticRegression

    X, y, treatment = make_dummy_data(n=200)
    y_wrong = y[:150]   # wrong length

    tl = CustomTLearner(LogisticRegression(), LogisticRegression())
    try:
        tl.fit(X, y_wrong, treatment)
        raise AssertionError("Should have raised ValueError for length mismatch")
    except ValueError:
        pass   # expected — this is the correct behaviour


def test_input_validation_non_binary_treatment():
    from custom_tlearner import CustomTLearner
    from sklearn.linear_model import LogisticRegression

    X, y, _ = make_dummy_data(n=200)
    bad_treatment = np.random.randint(0, 5, size=200)   # 0-4, not binary

    tl = CustomTLearner(LogisticRegression(), LogisticRegression())
    try:
        tl.fit(X, y, bad_treatment)
        raise AssertionError("Should have raised ValueError for non-binary treatment")
    except ValueError:
        pass   # expected


# =============================================================================
# Test 4: CustomTLearner works with XGBoost
# =============================================================================

def test_tlearner_with_xgboost():
    from custom_tlearner import CustomTLearner
    import xgboost as xgb

    X, y, treatment = make_dummy_data()

    xgb_clf = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    tl = CustomTLearner(treatment_model=xgb_clf, control_model=xgb_clf)
    tl.fit(X, y, treatment)

    uplift = tl.predict_uplift(X)
    assert uplift.shape == (200,)
    assert not np.any(np.isnan(uplift))


# =============================================================================
# Test 5: CustomTLearner works with EBM
# =============================================================================

def test_tlearner_with_ebm():
    from custom_tlearner import CustomTLearner
    from interpret.glassbox import ExplainableBoostingClassifier

    X, y, treatment = make_dummy_data()

    ebm = ExplainableBoostingClassifier(random_state=42, max_rounds=50)
    tl  = CustomTLearner(treatment_model=ebm, control_model=ebm)
    tl.fit(X, y, treatment)

    uplift = tl.predict_uplift(X)
    assert uplift.shape == (200,)
    assert not np.any(np.isnan(uplift))


# =============================================================================
# Test 6: Qini metric is importable and runs
# =============================================================================

def test_qini_metric():
    from custom_tlearner import CustomTLearner
    from sklearn.linear_model import LogisticRegression
    from sklift.metrics import qini_auc_score

    X, y, treatment = make_dummy_data()
    tl = CustomTLearner(LogisticRegression(max_iter=200), LogisticRegression(max_iter=200))
    tl.fit(X, y, treatment)
    uplift = tl.predict_uplift(X)

    score = qini_auc_score(y, uplift, treatment)
    assert isinstance(score, float), "AUQC must be a float"


# =============================================================================
# Runner
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Integration test — Causal Uplift EBM project")
    print("=" * 60)

    tests = [
        ("Config imports",                          test_config_imports),
        ("T-Learner + LogReg",                      test_tlearner_with_logreg),
        ("Input validation: length mismatch",       test_input_validation_length_mismatch),
        ("Input validation: non-binary treatment",  test_input_validation_non_binary_treatment),
        ("T-Learner + XGBoost",                     test_tlearner_with_xgboost),
        ("T-Learner + EBM",                         test_tlearner_with_ebm),
        ("Qini metric",                             test_qini_metric),
    ]

    results = [run_test(name, fn) for name, fn in tests]

    print("=" * 60)
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("All systems go. Safe to commit.")
    else:
        print("Fix failures before committing — do NOT push broken code.")
        sys.exit(1)   # non-zero exit code fails CI/CD pipelines