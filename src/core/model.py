"""
model.py
========
XMalDetect — Classifier wrappers for:
  - Random Forest (primary model)
  - SVM (baseline)
  - MLP / Neural Network (baseline)
  - RF with Gini Importance (baseline — no SHAP)

All wrappers expose a unified .train() / .predict() / .evaluate() API
so they can be swapped interchangeably in notebooks.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    cross_validate, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize


# ──────────────────────────────────────────────────────────────────────────────
# Base wrapper
# ──────────────────────────────────────────────────────────────────────────────

class BaseModel:
    """Shared interface for all classifiers used in XMalDetect."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        self.feature_names: list[str] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              feature_names: list[str] | None = None) -> None:
        """Fit the underlying sklearn estimator."""
        if feature_names is not None:
            self.feature_names = feature_names
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"[{self.name}] Training complete. "
              f"Samples: {len(y_train)} | "
              f"Features: {X_train.shape[1]}")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability estimates. Falls back to decision_function for SVM."""
        self._check_trained()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # SVM with probability=False — use decision function
        scores = self.model.decision_function(X)
        # Normalise to [0,1] with sigmoid
        proba_pos = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - proba_pos, proba_pos])

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 verbose: bool = True) -> dict:
        """Return a dictionary of all key metrics."""
        self._check_trained()
        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            "model"    : self.name,
            "accuracy" : accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall"   : recall_score(y_test, y_pred, zero_division=0),
            "f1"       : f1_score(y_test, y_pred, zero_division=0),
            "auc_roc"  : roc_auc_score(y_test, y_proba),
        }

        if verbose:
            print(f"\n{'='*55}")
            print(f"  Model : {self.name}")
            print(f"{'='*55}")
            for k, v in metrics.items():
                if k != "model":
                    print(f"  {k:<12}: {v:.4f}")
            print("\n  Classification Report:")
            print(classification_report(y_test, y_pred,
                                        target_names=["Benign", "Malware"]))

        return metrics

    # ── Cross-Validation ─────────────────────────────────────────────────────

    def cross_validate_report(self, X: np.ndarray, y: np.ndarray,
                               cv: int = 5) -> pd.DataFrame:
        """Run stratified k-fold CV and return mean ± std for each metric."""
        self._check_trained()
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        results = cross_validate(
            self.model, X, y,
            cv=skf, scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )
        summary = {}
        for s in scoring:
            vals = results[f"test_{s}"]
            summary[s] = f"{vals.mean():.4f} ± {vals.std():.4f}"

        df = pd.DataFrame([summary], index=[self.name])
        print(f"\n[{self.name}] {cv}-Fold CV Results:\n{df.T}")
        return df

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model,
                     "name": self.name,
                     "feature_names": self.feature_names}, path)
        print(f"[{self.name}] Saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.name          = data["name"]
        obj.model         = data["model"]
        obj.feature_names = data["feature_names"]
        obj.is_trained    = True
        print(f"[{obj.name}] Loaded <- {path}")
        return obj

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError(
                f"[{self.name}] Model has not been trained yet. "
                "Call .train() first."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Random Forest (primary model)
# ──────────────────────────────────────────────────────────────────────────────

class RandomForestModel(BaseModel):
    """
    Random Forest classifier with SHAP-compatible TreeExplainer support.

    Why RF over deep learning?
    - TreeSHAP computes EXACT Shapley values in polynomial time.
    - Alajmani et al. (IJNSA 2025) show RF outperforms CNN & DNN on
      structured Android permission features (69.3% vs 59.5%/59.2%).
    - Resistant to overfitting on high-dimensional binary feature spaces.
    """

    def __init__(
        self,
        n_estimators:      int   = 200,
        max_depth:         int | None = None,
        min_samples_split: int   = 2,
        min_samples_leaf:  int   = 1,
        max_features:      str   = "sqrt",
        class_weight:      str   = "balanced",   # handles class imbalance
        random_state:      int   = 42,
        n_jobs:            int   = -1,
    ):
        super().__init__("Random Forest + SHAP")
        self.model = RandomForestClassifier(
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf  = min_samples_leaf,
            max_features      = max_features,
            class_weight      = class_weight,
            random_state      = random_state,
            n_jobs            = n_jobs,
        )
        self.params = {
            "n_estimators"     : n_estimators,
            "max_depth"        : max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf" : min_samples_leaf,
            "max_features"     : max_features,
            "class_weight"     : class_weight,
            "random_state"     : random_state,
        }

    # ── Hyperparameter tuning ─────────────────────────────────────────────────

    def tune(self, X_train: np.ndarray, y_train: np.ndarray,
             cv: int = 3, verbose: int = 1) -> dict:
        """
        Grid search over key hyperparameters.
        Returns the best params dict and updates self.model.
        """
        param_grid = {
            "n_estimators"     : [100, 200, 300],
            "max_depth"        : [None, 10, 20],
            "min_samples_split": [2, 5],
            "max_features"     : ["sqrt", "log2"],
        }
        gs = GridSearchCV(
            self.model, param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring="f1",
            n_jobs=-1,
            verbose=verbose,
        )
        gs.fit(X_train, y_train)
        self.model = gs.best_estimator_
        self.is_trained = True
        print(f"\n[RF] Best params: {gs.best_params_}")
        print(f"[RF] Best CV F1:   {gs.best_score_:.4f}")
        return gs.best_params_

    # ── Gini feature importance (baseline comparison) ─────────────────────────

    def gini_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Return top_n features by Gini (MDI) importance.
        Use this to compare against SHAP in Section 6.5 of the paper.
        """
        self._check_trained()
        importances = self.model.feature_importances_
        names = (self.feature_names if self.feature_names
                 else [f"f_{i}" for i in range(len(importances))])
        df = (pd.DataFrame({"feature": names, "gini_importance": importances})
              .sort_values("gini_importance", ascending=False)
              .head(top_n)
              .reset_index(drop=True))
        df.index += 1  # 1-based rank
        return df


# ──────────────────────────────────────────────────────────────────────────────
# SVM Baseline (black-box, no explanations)
# ──────────────────────────────────────────────────────────────────────────────

class SVMModel(BaseModel):
    """
    Support Vector Machine — black-box baseline.
    Provides no feature explanations; included to demonstrate the
    accuracy-vs-explainability trade-off.
    """

    def __init__(
        self,
        C:            float = 1.0,
        kernel:       str   = "rbf",
        gamma:        str   = "scale",
        class_weight: str   = "balanced",
        probability:  bool  = True,   # needed for predict_proba + ROC-AUC
        random_state: int   = 42,
    ):
        super().__init__("SVM (Black-Box Baseline)")
        self.model = SVC(
            C            = C,
            kernel       = kernel,
            gamma        = gamma,
            class_weight = class_weight,
            probability  = probability,
            random_state = random_state,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MLP Baseline (black-box, no explanations)
# ──────────────────────────────────────────────────────────────────────────────

class MLPModel(BaseModel):
    """
    Multi-Layer Perceptron — shallow neural network black-box baseline.
    Demonstrates that adding layers does NOT equal adding interpretability.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (256, 128, 64),
        activation:         str   = "relu",
        solver:             str   = "adam",
        alpha:              float = 1e-4,
        max_iter:           int   = 300,
        random_state:       int   = 42,
    ):
        super().__init__("MLP Neural Network (Black-Box Baseline)")
        self.model = MLPClassifier(
            hidden_layer_sizes = hidden_layer_sizes,
            activation         = activation,
            solver             = solver,
            alpha              = alpha,
            max_iter           = max_iter,
            random_state       = random_state,
        )


# ──────────────────────────────────────────────────────────────────────────────
# RF with Gini Importance only (partial explainability baseline)
# ──────────────────────────────────────────────────────────────────────────────

class RFGiniModel(RandomForestModel):
    """
    Same Random Forest as the primary model, but tracked separately
    to represent the 'RF without SHAP' baseline in comparison tables.
    Gini importance is global (one ranking for all samples); SHAP is
    per-sample — that is the key differentiator demonstrated in §6.5.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RF + Gini Importance (Partial Baseline)"


# ──────────────────────────────────────────────────────────────────────────────
# Comparison runner
# ──────────────────────────────────────────────────────────────────────────────

class ModelComparison:
    """
    Trains all four models on the same data and collects results into
    a single comparison DataFrame for Table 3 in the paper.
    """

    def __init__(self):
        self.models: list[BaseModel] = [
            RandomForestModel(),
            RFGiniModel(),
            SVMModel(),
            MLPModel(),
        ]
        self.results: list[dict] = []

    def run(self, X_train, y_train, X_test, y_test,
            feature_names: list[str] | None = None,
            cv: int = 5) -> pd.DataFrame:
        """
        Train all models, evaluate, cross-validate,
        and return a summary DataFrame.
        """
        self.results = []
        for m in self.models:
            print(f"\n{'─'*55}")
            print(f"  Training: {m.name}")
            print(f"{'─'*55}")
            m.train(X_train, y_train, feature_names)
            metrics = m.evaluate(X_test, y_test, verbose=True)
            cv_df   = m.cross_validate_report(
                np.vstack([X_train, X_test]),
                np.hstack([y_train, y_test]),
                cv=cv
            )
            metrics["cv_f1"] = cv_df["f1"].values[0]
            metrics["explainable"] = "Full (SHAP)" if "SHAP" in m.name \
                else ("Partial (Gini)" if "Gini" in m.name else "None")
            self.results.append(metrics)

        df = pd.DataFrame(self.results).set_index("model")
        print(f"\n{'='*70}")
        print("  MODEL COMPARISON TABLE (Table 3 — Paper)")
        print(f"{'='*70}")
        print(df.to_string())
        return df

    def save_results(self, path: str = "outputs/comparison_table.csv") -> None:
        df = pd.DataFrame(self.results)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Comparison table saved -> {path}")