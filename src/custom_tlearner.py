# =============================================================================
# custom_tlearner.py  —  Member 2: Meta-Learner Architect
# -----------------------------------------------------------------------------
# Implements the T-Learner (Two-Model Learner) from scratch using
# scikit-learn's BaseEstimator interface.
#
# WHAT IS A T-LEARNER?
#   Standard ML predicts P(Y=1 | X) for everyone.  That answers:
#   "Who is likely to buy?"  But we want to answer:
#   "Who will buy BECAUSE of the ad?" — a fundamentally different question.
#
#   The T-Learner answers it by training two completely separate models:
#       μ₁(X) = P(Y=1 | X, T=1)  — trained ONLY on treated users
#       μ₀(X) = P(Y=1 | X, T=0)  — trained ONLY on control users
#
#   The Conditional Average Treatment Effect (CATE) / uplift for a user is:
#       τ(X) = μ₁(X) − μ₀(X)
#
#   Positive τ(X) → the ad increases this user's conversion probability.
#   Negative τ(X) → the ad HURTS this user ("sleeping dog").
#   Near-zero τ(X) → the user is unaffected ("sure thing" or "lost cause").
#
# WHY BUILD IT OURSELVES (not use EconML / CausalML)?
#   Those libraries are black boxes.  Building it from scratch using
#   sklearn's BaseEstimator proves we understand the underlying math —
#   which is exactly what the viva panel will test.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone


class CustomTLearner(BaseEstimator):
    """
    T-Learner meta-model for causal uplift estimation.

    Wraps any two sklearn-compatible classifiers (must have predict_proba)
    and routes training data to each based on the treatment flag.

    Parameters
    ----------
    treatment_model : sklearn classifier
        The model μ₁ — trained on treated users (T=1).
        Must implement fit() and predict_proba().
        Example: ExplainableBoostingClassifier() or XGBClassifier()

    control_model : sklearn classifier
        The model μ₀ — trained on control users (T=0).
        Can be a different class or hyperparameters from treatment_model,
        but in practice we use the same architecture for comparability.

    Example
    -------
        from src.custom_tlearner import CustomTLearner
        from interpret.glassbox import ExplainableBoostingClassifier

        ebm = ExplainableBoostingClassifier(random_state=42)
        tlearner = CustomTLearner(treatment_model=ebm, control_model=ebm)
        tlearner.fit(X_train, y_train, t_train)
        uplift_scores = tlearner.predict_uplift(X_test)
    """

    def __init__(self, treatment_model, control_model):
        # ── scikit-learn rule: store constructor args with IDENTICAL names ──
        # Do NOT add any logic here.  sklearn's get_params() / clone() rely
        # on __init__ parameter names matching attribute names exactly.
        self.treatment_model = treatment_model
        self.control_model   = control_model

    # =========================================================================
    # fit()  —  trains μ₁ and μ₀ on their respective subgroups
    # =========================================================================

    def fit(self, X, y, treatment):
        """
        Trains two sub-models by routing data based on the treatment flag.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Feature matrix for ALL users (treated + control combined).

        y : array-like, shape (n_samples,)
            Binary target variable (0 or 1) for all users.

        treatment : array-like, shape (n_samples,)
            Binary treatment flag: 1 = received ad, 0 = control group.

        Returns
        -------
        self  (sklearn convention — enables method chaining)
        """

        # ── Input validation ──────────────────────────────────────────────────
        X, y, treatment = self._validate_inputs(X, y, treatment)

        # ── Clone base models so we start fresh every time fit() is called ───
        # clone() creates a new, UNFITTED copy with the same hyperparameters.
        # Without clone(), calling fit() twice would retrain on stale state.
        self.model_t1_ = clone(self.treatment_model)  # trailing _ = fitted attr
        self.model_t0_ = clone(self.control_model)

        # ── Data routing: split by treatment flag ─────────────────────────────
        # Boolean mask: True where user received the ad
        treated_mask  = (treatment == 1)
        control_mask  = (treatment == 0)

        # Subset feature matrices and targets for each group
        X_treated = X[treated_mask]
        y_treated = y[treated_mask]

        X_control = X[control_mask]
        y_control = y[control_mask]

        print(f"[T-Learner] Fitting μ₁ on {len(X_treated):,} treated users...")
        self.model_t1_.fit(X_treated, y_treated)

        print(f"[T-Learner] Fitting μ₀ on {len(X_control):,} control users...")
        self.model_t0_.fit(X_control, y_control)

        print("[T-Learner] Both sub-models fitted successfully.")
        return self   # return self so sklearn pipelines work correctly

    # =========================================================================
    # predict_uplift()  —  computes τ(X) = μ₁(X) − μ₀(X)
    # =========================================================================

    def predict_uplift(self, X):
        """
        Computes the Conditional Average Treatment Effect (CATE) / uplift
        for each user in X.

        Mathematically:  τ(X) = μ₁(X) − μ₀(X)

        Both μ₁ and μ₀ predict the PROBABILITY of the positive outcome.
        Their difference is the *incremental* probability caused by the ad.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Feature matrix.  Does NOT need treatment flag — we are doing
            inference now, not training.

        Returns
        -------
        tau : np.ndarray, shape (n_samples,)
            Uplift scores.  Values range roughly from -1 to +1.
            Positive  → ad is predicted to INCREASE conversion probability.
            Negative  → ad is predicted to DECREASE it (sleeping dog).
            Near zero → ad has little effect.
        """
        # Convert to numpy for consistent indexing
        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)

        # μ₁(X): probability of positive outcome UNDER treatment
        # predict_proba returns [[P(y=0), P(y=1)], ...] — we want column 1
        mu1 = self.model_t1_.predict_proba(X_arr)[:, 1]

        # μ₀(X): probability of positive outcome WITHOUT treatment
        mu0 = self.model_t0_.predict_proba(X_arr)[:, 1]

        # τ(X) = the causal uplift — the core output of this entire project
        tau = mu1 - mu0

        return tau

    # =========================================================================
    # shap_values_diff()  —  SHAP attribution at the causal level
    # =========================================================================

    def shap_values_diff(self, X, background_sample_size: int = 100):
        """
        Computes SHAP values for the uplift τ(X) by running TreeExplainer
        separately on μ₁ and μ₀, then taking the difference.

        WHY THIS APPROACH OVER KernelExplainer ON predict_uplift:
            KernelExplainer treats the whole T-Learner as a black box,
            sampling the function many times — slow and approximate.

            This method runs TreeExplainer (exact, fast, model-aware) on
            each EBM/XGBoost sub-model separately, then computes:
                SHAP_uplift(X) = SHAP_μ₁(X) − SHAP_μ₀(X)

            The result tells us: "How much did each feature contribute to
            pushing the UPLIFT higher or lower?" — attribution at the
            causal level, not just the predictive level.

            This is methodologically stronger and is publishable quality.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix (use the test set or a representative sample).

        background_sample_size : int
            Only used if TreeExplainer cannot be used (falls back to
            KernelExplainer with kmeans background).

        Returns
        -------
        shap_diff : np.ndarray, shape (n_samples, n_features)
            SHAP values for the uplift prediction per user per feature.

        Example (in notebook):
            shap_diff = tlearner.shap_values_diff(X_test)
            import shap
            shap.summary_plot(shap_diff, X_test)
        """
        import shap

        X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else np.array(X)

        try:
            # TreeExplainer works directly with tree-based models (EBM, XGB)
            # It is exact (no sampling approximation) and very fast.
            explainer_t1 = shap.TreeExplainer(self.model_t1_)
            explainer_t0 = shap.TreeExplainer(self.model_t0_)

            # SHAP values for the positive class (index 1)
            shap_t1 = explainer_t1.shap_values(X_arr)
            shap_t0 = explainer_t0.shap_values(X_arr)

            # Handle the case where shap_values returns a list [class0, class1]
            if isinstance(shap_t1, list):
                shap_t1 = shap_t1[1]
            if isinstance(shap_t0, list):
                shap_t0 = shap_t0[1]

        except Exception as e:
            # Fallback: KernelExplainer for models that TreeExplainer can't handle
            print(f"[SHAP] TreeExplainer failed ({e}). Falling back to KernelExplainer.")
            print("       This is slower — be patient.")

            background = shap.kmeans(X_arr, background_sample_size)

            def predict_fn_t1(x): return self.model_t1_.predict_proba(x)[:, 1]
            def predict_fn_t0(x): return self.model_t0_.predict_proba(x)[:, 1]

            explainer_t1 = shap.KernelExplainer(predict_fn_t1, background)
            explainer_t0 = shap.KernelExplainer(predict_fn_t0, background)

            shap_t1 = explainer_t1.shap_values(X_arr)
            shap_t0 = explainer_t0.shap_values(X_arr)

        # The causal SHAP attribution: difference of the two SHAP matrices
        shap_diff = shap_t1 - shap_t0

        print(f"[SHAP] shap_diff shape: {shap_diff.shape}  (users × features)")
        return shap_diff

    # =========================================================================
    # _validate_inputs()  —  private helper, called inside fit()
    # =========================================================================

    def _validate_inputs(self, X, y, treatment):
        """
        Validates that inputs are consistent and safe before training begins.
        Raises clear errors so bugs are caught early, not mid-training.
        """

        # Convert everything to numpy for consistent handling
        if isinstance(X, pd.DataFrame):
            X_out = X.reset_index(drop=True)   # avoid pandas index bugs
        else:
            X_out = np.array(X)

        y_out = np.array(y)
        t_out = np.array(treatment)

        # ── Length consistency ────────────────────────────────────────────────
        n = len(X_out)
        if len(y_out) != n or len(t_out) != n:
            raise ValueError(
                f"X, y, and treatment must all have the same length. "
                f"Got X={n}, y={len(y_out)}, treatment={len(t_out)}."
            )

        # ── Treatment must be strictly binary ─────────────────────────────────
        unique_t = set(np.unique(t_out))
        if not unique_t.issubset({0, 1}):
            raise ValueError(
                f"treatment must contain only 0 and 1. "
                f"Found values: {unique_t}. "
                "Did you forget to binarise the segment column?"
            )

        # ── Both groups must have enough data ─────────────────────────────────
        n_treated = t_out.sum()
        n_control = (t_out == 0).sum()
        if n_treated < 50 or n_control < 50:
            raise ValueError(
                f"Too few samples: treated={n_treated}, control={n_control}. "
                "Need at least 50 in each group to train a meaningful model."
            )

        # ── Base models must support predict_proba ────────────────────────────
        for name, model in [("treatment_model", self.treatment_model),
                             ("control_model",   self.control_model)]:
            if not hasattr(model, "predict_proba"):
                raise AttributeError(
                    f"{name} ({type(model).__name__}) does not have predict_proba(). "
                    "The T-Learner requires probability outputs to compute τ(X) = μ₁ − μ₀. "
                    "Use a classifier, not a regressor."
                )

        return X_out, y_out, t_out