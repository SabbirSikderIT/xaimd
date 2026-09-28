"""
shap_engine.py
==============
XMalDetect — SHAP Explanation Engine

Wraps shap.TreeExplainer for the Random Forest model and produces
all three explanation outputs described in §5.3 of the paper:

  Output A — Global summary plot (top-N features across all test samples)
  Output B — Per-sample waterfall / force plot (forensic evidence map)
  Output C — Malware-family SHAP fingerprint heat map

Key advantage over LIME:
  TreeSHAP computes EXACT Shapley values in polynomial time.
  LIME produces stochastic approximations. Shapley values satisfy
  the local accuracy, consistency, and missingness axioms (Lundberg & Lee, 2017).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # non-interactive backend for server / CI
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
except ImportError:
    raise ImportError(
        "shap is not installed. Run: pip install shap>=0.44.0"
    )

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (consistent across all figures)
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "malware" : "#d62728",   # red
    "benign"  : "#1f77b4",   # blue
    "neutral" : "#7f7f7f",   # grey
    "accent"  : "#ff7f0e",   # orange
}


# ──────────────────────────────────────────────────────────────────────────────
# SHAPEngine
# ──────────────────────────────────────────────────────────────────────────────

class SHAPEngine:
    """
    Main SHAP wrapper.  Operates on a trained RandomForestModel instance.

    Usage
    -----
    >>> engine = SHAPEngine(rf_model, feature_names=feature_names)
    >>> engine.fit(X_test)
    >>> engine.plot_global_summary(save_path="outputs/figures/fig3_shap_summary.png")
    >>> engine.plot_waterfall(sample_idx=0, save_path="outputs/figures/fig4_waterfall_0.png")
    >>> engine.plot_family_heatmap(X_test, y_pred, save_path="outputs/figures/fig5_family_heatmap.png")
    """

    def __init__(self, rf_model, feature_names: list[str] | None = None):
        """
        Parameters
        ----------
        rf_model     : trained RandomForestModel (or any sklearn RF estimator)
        feature_names: list of feature name strings (Android permissions)
        """
        sklearn_model = (rf_model.model
                         if hasattr(rf_model, "model") else rf_model)

        self.explainer     = shap.TreeExplainer(sklearn_model)
        self.feature_names = (feature_names if feature_names is not None
                              else [f"feature_{i}"
                                    for i in range(sklearn_model.n_features_in_)])

        self.shap_values: np.ndarray | None = None   # shape (n_samples, n_features)
        self.X_test: np.ndarray | None      = None
        self.base_value: float | None       = None
        self.n_features                     = len(self.feature_names)

    # ── Compute SHAP values ───────────────────────────────────────────────────

    def fit(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for all samples in X_test.
        For binary classification, returns values for the MALWARE class (index 1).

        Returns
        -------
        shap_values : np.ndarray, shape (n_samples, n_features)
        """
        print(f"[SHAP] Computing TreeSHAP values for {len(X_test)} samples ...")
        raw = self.explainer.shap_values(X_test)

        if isinstance(raw, np.ndarray) and raw.ndim == 3:
            self.shap_values = raw[:, :, 1]
            ev = self.explainer.expected_value
            self.base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)
        elif isinstance(raw, list):
            self.shap_values = raw[1]
            self.base_value = self.explainer.expected_value[1]
        else:
            self.shap_values = raw
            ev = self.explainer.expected_value
            self.base_value = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)

        self.X_test = X_test
        print(f"[SHAP] Done. SHAP matrix shape: {self.shap_values.shape}")
        return self.shap_values

    # ── Helper: mean absolute SHAP by feature ─────────────────────────────────

    def mean_abs_shap(self, top_n: int = 20) -> pd.DataFrame:
        """Return top_n features ranked by mean |SHAP| value."""
        self._check_fitted()
        mean_abs = np.mean(np.abs(self.shap_values), axis=0)
        df = pd.DataFrame({
            "feature"     : self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
        df.index += 1   # 1-based rank
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT A — Global Summary Plot  (Figure 3 in paper)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_global_summary(
        self,
        top_n:     int = 20,
        plot_type: str = "dot",     # "dot" = beeswarm | "bar" = bar chart
        save_path: str = "outputs/figures/fig3_shap_summary.png",
        dpi:       int = 300,
    ) -> None:
        """
        Global SHAP summary plot showing the top-N most important features
        across all test samples.

        plot_type='dot'  -> beeswarm (shows distribution + direction)
        plot_type='bar'  -> bar chart (mean |SHAP|, simpler for papers)
        """
        self._check_fitted()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Use shap's built-in Explanation object for newer API
        explanation = shap.Explanation(
            values        = self.shap_values,
            base_values   = np.full(len(self.shap_values), self.base_value),
            data          = self.X_test,
            feature_names = self.feature_names,
        )

        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(
            explanation,
            max_display = top_n,
            show        = False,
        ) if plot_type == "dot" else shap.plots.bar(
            explanation,
            max_display = top_n,
            show        = False,
        )

        plt.title(
            f"Fig. 3 — SHAP Global Feature Importance (Top {top_n})\n"
            "Red = pushes toward Malware | Blue = pushes toward Benign",
            fontsize=11, pad=12
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Global summary saved -> {save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT B — Per-Sample Waterfall Plot  (Figure 4 in paper)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_waterfall(
        self,
        sample_idx: int = 0,
        top_n:      int = 15,
        save_path:  str | None = None,
        dpi:        int = 300,
        title_suffix: str = "",
    ) -> None:
        """
        Waterfall chart for a single sample — the core 'Forensic Evidence Map'.

        Shows which permissions pushed the prediction toward Malware (+, red)
        and which pushed toward Benign (-, blue), with the base value as origin.

        Parameters
        ----------
        sample_idx   : index of sample in X_test
        top_n        : number of top-contributing features to show
        save_path    : if None, auto-generates path with sample index
        title_suffix : e.g. "— Case Study 1: Spyware (Confirmed)"
        """
        self._check_fitted()
        if save_path is None:
            save_path = f"outputs/figures/fig4_waterfall_sample{sample_idx}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        exp = shap.Explanation(
            values        = self.shap_values[sample_idx],
            base_values   = self.base_value,
            data          = self.X_test[sample_idx],
            feature_names = self.feature_names,
        )

        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(exp, max_display=top_n, show=False)

        fig_title = (f"Fig. 4 — SHAP Forensic Evidence Map "
                     f"(Sample #{sample_idx})")
        if title_suffix:
            fig_title += f"\n{title_suffix}"
        plt.title(fig_title, fontsize=11, pad=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Waterfall chart saved -> {save_path}")

    def plot_force(
        self,
        sample_idx: int = 0,
        save_path:  str | None = None,
        dpi:        int = 300,
    ) -> None:
        """
        SHAP force plot (horizontal) — alternative to waterfall.
        Shows the push-pull balance of all features in one line.
        """
        self._check_fitted()
        if save_path is None:
            save_path = f"outputs/figures/force_sample{sample_idx}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        shap.force_plot(
            base_value    = self.base_value,
            shap_values   = self.shap_values[sample_idx],
            features      = self.X_test[sample_idx],
            feature_names = self.feature_names,
            matplotlib    = True,
            show          = False,
        )
        plt.title(f"SHAP Force Plot — Sample #{sample_idx}", pad=20)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Force plot saved -> {save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT C — Malware Family Fingerprint Heat Map  (Figure 5 in paper)
    # ─────────────────────────────────────────────────────────────────────────

    def plot_family_heatmap(
        self,
        y_pred:     np.ndarray,
        y_families: np.ndarray | None = None,
        top_n:      int = 20,
        save_path:  str = "outputs/figures/fig5_family_heatmap.png",
        dpi:        int = 300,
        cmap:       str = "RdBu_r",
    ) -> pd.DataFrame:
        """
        For each malware family, compute mean SHAP value per feature and
        display as a heat map.  This is the 'SHAP fingerprint' that
        differentiates your work from Basheer et al. (2024).

        Parameters
        ----------
        y_pred     : predicted labels (0=Benign, 1=Malware)
        y_families : array of family labels (e.g., 'Spyware', 'Adware').
                     If None, uses generic 'Malware' vs 'Benign' grouping.

        Returns
        -------
        family_df : DataFrame of mean SHAP per family (rows=families, cols=features)
        """
        self._check_fitted()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if y_families is None:
            # Fall back to predicted binary labels
            y_families = np.where(y_pred == 1, "Malware", "Benign")

        families = np.unique(y_families)
        family_means = {}
        for fam in families:
            mask = y_families == fam
            if mask.sum() == 0:
                continue
            family_means[fam] = np.mean(self.shap_values[mask], axis=0)

        family_df = pd.DataFrame(family_means, index=self.feature_names)

        # Select top_n features by overall mean |SHAP|
        top_features = (np.abs(self.shap_values).mean(axis=0)
                        .argsort()[::-1][:top_n])
        plot_df = family_df.iloc[top_features]

        fig, ax = plt.subplots(figsize=(max(8, len(families) * 2), 10))
        sns.heatmap(
            plot_df,
            cmap       = cmap,
            center     = 0,
            annot      = True,
            fmt        = ".3f",
            linewidths = 0.5,
            ax         = ax,
            cbar_kws   = {"label": "Mean SHAP Value"},
        )
        ax.set_title(
            f"Fig. 5 — SHAP Fingerprint per Malware Family (Top {top_n} Features)\n"
            "Positive (red) = pushes toward class | Negative (blue) = pushes away",
            fontsize=11, pad=12
        )
        ax.set_xlabel("Malware Family", fontsize=11)
        ax.set_ylabel("Android Permission / Feature", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Family heatmap saved -> {save_path}")
        return family_df

    # ─────────────────────────────────────────────────────────────────────────
    # SHAP vs Gini Importance Comparison  (§6.5 in paper)
    # ─────────────────────────────────────────────────────────────────────────

    def compare_with_gini(
        self,
        gini_df:   pd.DataFrame,
        top_n:     int = 15,
        save_path: str = "outputs/figures/fig_shap_vs_gini.png",
        dpi:       int = 300,
    ) -> None:
        """
        Side-by-side bar comparison of SHAP ranking vs Gini ranking.
        Demonstrates that SHAP provides different (more nuanced) rankings
        than traditional Gini importance.

        Parameters
        ----------
        gini_df : output of RandomForestModel.gini_importance()
        """
        self._check_fitted()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        shap_df = self.mean_abs_shap(top_n=top_n)
        gini_top = gini_df.head(top_n)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # SHAP side
        axes[0].barh(
            shap_df["feature"][::-1],
            shap_df["mean_abs_shap"][::-1],
            color=PALETTE["malware"], alpha=0.8
        )
        axes[0].set_title("SHAP Feature Ranking\n(Mean |SHAP|)", fontsize=12)
        axes[0].set_xlabel("Mean |SHAP Value|")

        # Gini side
        axes[1].barh(
            gini_top["feature"][::-1],
            gini_top["gini_importance"][::-1],
            color=PALETTE["benign"], alpha=0.8
        )
        axes[1].set_title("Gini Feature Ranking\n(Mean Decrease Impurity)", fontsize=12)
        axes[1].set_xlabel("Gini Importance")

        fig.suptitle(
            "SHAP vs. Gini Importance — Feature Rankings Comparison (§6.5)\n"
            "SHAP is per-sample; Gini is global — demonstrating SHAP's superiority",
            fontsize=11, y=1.01
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] SHAP vs Gini comparison saved -> {save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Export SHAP values as DataFrame
    # ─────────────────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """Return SHAP values as a labelled DataFrame."""
        self._check_fitted()
        return pd.DataFrame(self.shap_values, columns=self.feature_names)

    def top_features_for_sample(self, sample_idx: int,
                                 top_n: int = 10) -> pd.DataFrame:
        """Return top contributing features for a single sample — for case studies."""
        self._check_fitted()
        vals  = self.shap_values[sample_idx]
        feats = self.feature_names
        df = pd.DataFrame({
            "feature"    : feats,
            "shap_value" : vals,
            "direction"  : ["> Malware" if v > 0 else "> Benign" for v in vals],
        })
        df["abs_shap"] = df["shap_value"].abs()
        return (df.sort_values("abs_shap", ascending=False)
                  .head(top_n)
                  .drop(columns="abs_shap")
                  .reset_index(drop=True))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self.shap_values is None:
            raise RuntimeError(
                "[SHAP] No SHAP values computed yet. Call .fit(X_test) first."
            )