"""Tree-based explainers using native importance, decision paths, and SHAP."""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.tree import export_text

from explainability.base_explainer import BaseExplainer, ExplanationResult
from explainability.explainers.model_agnostic_explainer import ModelAgnosticExplainer
from explainability.visualizations import (
    plot_native_feature_importance,
    plot_permutation_importance,
    plot_shap_beeswarm,
    plot_shap_global_bar,
    plot_shap_waterfall,
)
from modeling_utils import DEFAULT_RANDOM_STATE


class TreeShapMixin:
    """Provide SHAP helpers shared by tree-based explainers."""

    if TYPE_CHECKING:
        config: dict[str, Any]
        pipeline: Any

        @property
        def classifier(self) -> Any:
            """Return the classifier step from the wrapped pipeline."""

            ...

        def _as_dataframe(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame: ...

        def _preprocess_for_classifier(
            self, x: pd.DataFrame | np.ndarray
        ) -> np.ndarray: ...

        def _model_feature_names(self, x: pd.DataFrame | np.ndarray) -> list[str]: ...

        def _configured_positive_class_index(self) -> int | None: ...

    def _shap_global(self, x: pd.DataFrame) -> dict[str, Any] | None:
        """Compute sampled global SHAP values for tree classifiers."""

        try:
            import shap
        except ImportError:
            return None

        sample_size = int(self.config.get("shap_sample_size", min(len(x), 100)))
        x_sample = self._as_dataframe(x).sample(
            n=min(sample_size, len(x)),
            random_state=int(self.config.get("random_state", DEFAULT_RANDOM_STATE)),
        )

        z_sample = self._preprocess_for_classifier(x_sample)
        feature_names = self._model_feature_names(x_sample)
        z_sample_df = pd.DataFrame(z_sample, columns=feature_names)

        explainer = shap.TreeExplainer(self.classifier)
        shap_values = explainer(z_sample_df)
        class_index = self._class_index_for_shap()
        selected = self._select_shap_class(shap_values, class_index=class_index)
        values = np.asarray(selected.values)
        mean_abs = np.abs(values).mean(axis=0)

        return {
            "values": selected,
            "sample": z_sample_df,
            "explained_class": self._class_label_for_shap_index(class_index),
            "explained_class_index": class_index,
            "mean_abs_shap": pd.DataFrame(
                {"feature": feature_names, "mean_abs_shap": mean_abs}
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True),
        }

    def _shap_local(
        self, x: pd.DataFrame, instance_index: int
    ) -> dict[str, Any] | None:
        """Compute SHAP values for one input instance."""

        try:
            import shap
        except ImportError:
            return None

        x_df = self._as_dataframe(x)
        x_row = x_df.iloc[[instance_index]]

        z_row = self._preprocess_for_classifier(x_row)
        feature_names = self._model_feature_names(x_df)
        z_row_df = pd.DataFrame(z_row, columns=feature_names)

        explainer = shap.TreeExplainer(self.classifier)
        shap_values = explainer(z_row_df)
        class_index = self._class_index_for_shap()
        selected = self._select_shap_class(shap_values, class_index=class_index)

        return {
            "values": selected[0] if len(selected.values.shape) == 2 else selected,
            "instance": z_row_df,
            "explained_class": self._class_label_for_shap_index(class_index),
            "explained_class_index": class_index,
        }

    def _select_shap_class(
        self, shap_values: Any, class_index: int | None = None
    ) -> Any:
        """Select the configured or predicted class from multiclass SHAP output."""

        values = np.asarray(shap_values.values)
        if values.ndim != 3:
            return shap_values

        if class_index is None:
            class_index = self._class_index_for_shap()
        return shap_values[:, :, class_index]

    def _class_index_for_shap(self) -> int:
        """Return the class index to use for class-specific SHAP explanations."""

        class_index = self._configured_positive_class_index()
        if class_index is None:
            raise ValueError(
                "Class-specific SHAP explanations require an explicit "
                "explainability_config['positive_class']. Do not rely on "
                "classifier.classes_[1] to define the explained class."
            )
        return class_index

    def _class_label_for_shap_index(self, class_index: int) -> Any:
        """Return the classifier class label for a SHAP class index."""

        classes = list(getattr(self.classifier, "classes_", []))
        if 0 <= class_index < len(classes):
            return classes[class_index]
        return self.config.get("positive_class")


class DecisionTreeExplainer(TreeShapMixin, BaseExplainer):
    """Explain decision trees with rules, native importance, paths, and SHAP."""

    def explain_global(
        self,
        x: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> ExplanationResult:
        """Generate global explanations for a fitted decision tree."""

        feature_names = self._model_feature_names(x)

        export_kwargs: dict[str, Any] = {
            "feature_names": feature_names,
            "decimals": int(self.config.get("tree_decimals", 4)),
        }

        if self.config.get("tree_max_depth") is not None:
            export_kwargs["max_depth"] = int(self.config["tree_max_depth"])

        native = self._native_feature_importance_frame(x)
        artifacts: dict[str, Any] = {
            **self._explanation_metadata(),
            "tree_rules": export_text(self.classifier, **export_kwargs),
            "feature_importance": native,
            "tree_depth": self.classifier.get_depth(),
            "n_leaves": self.classifier.get_n_leaves(),
        }

        warnings = self._common_warnings()
        figures: dict[str, str] = {
            "native_feature_importance": plot_native_feature_importance(
                native,
                output_dir=self.config.get("output_dir", "reports"),
                top_n=int(self.config.get("top_n_features", 15)),
            )
        }
        tables: dict[str, str] = {
            "native_feature_importance": self.save_table(
                native,
                "native_feature_importance.csv",
            )
        }
        tree_rules_path = self.get_output_dir("tables") / "tree_rules.txt"
        tree_rules_path.write_text(artifacts["tree_rules"], encoding="utf-8")
        tables["tree_rules"] = str(tree_rules_path)

        if y is not None:
            permutation = self._permutation_importance_frame(x, y)
            artifacts["permutation_importance"] = permutation
            tables["permutation_importance"] = self.save_table(
                permutation,
                "permutation_importance.csv",
            )
            figures["permutation_importance"] = plot_permutation_importance(
                permutation,
                output_dir=self.config.get("output_dir", "reports"),
                top_n=int(self.config.get("top_n_features", 15)),
            )

        shap_result = self._shap_global(x)
        if shap_result is not None:
            artifacts["shap_global"] = shap_result
            figures["shap_global_bar"] = plot_shap_global_bar(
                shap_result["values"],
                output_dir=self.config.get("output_dir", "reports"),
            )
            figures["shap_beeswarm"] = plot_shap_beeswarm(
                shap_result["values"],
                output_dir=self.config.get("output_dir", "reports"),
            )
        else:
            warnings.append(
                "SHAP global explanation was not generated. Install shap to enable "
                "SHAP artifacts."
            )

        return ExplanationResult(
            model_type=type(self.classifier).__name__,
            scope="global",
            artifacts=artifacts,
            warnings=warnings,
            figures=figures,
            tables=tables,
        )

    def explain_local(self, x: pd.DataFrame, instance_index: int) -> ExplanationResult:
        """Generate a local decision-path explanation for one instance."""

        x_df = self._as_dataframe(x)
        x_row = x_df.iloc[[instance_index]]
        z = self._preprocess_for_classifier(x_row)

        tree = self.classifier.tree_
        feature_names = self._model_feature_names(x_df)

        node_indicator = self.classifier.decision_path(z)
        leaf_id = self.classifier.apply(z)[0]

        node_index = node_indicator.indices[
            node_indicator.indptr[0] : node_indicator.indptr[1]
        ]

        conditions: list[dict[str, Any]] = []

        for node_id in node_index:
            if node_id == leaf_id:
                continue

            feature_index = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            value = z[0, feature_index]

            operator = "<=" if value <= threshold else ">"

            conditions.append(
                {
                    "node_id": int(node_id),
                    "feature": feature_names[feature_index],
                    "value_after_preprocessing": float(value),
                    "operator": operator,
                    "threshold": float(threshold),
                }
            )

        decision_path_frame = pd.DataFrame(conditions)
        artifacts = {
            **self._explanation_metadata(),
            "prediction": self._predict_summary(x_row),
            "leaf_id": int(leaf_id),
            "decision_path": decision_path_frame,
        }
        tables = {
            "decision_path": self.save_table(
                decision_path_frame,
                f"decision_path_instance_{instance_index}.csv",
            )
        }
        figures: dict[str, str] = {}
        shap_local = self._shap_local(x_df, instance_index)
        warnings = self._common_warnings() + self._consume_probability_warnings()
        if shap_local is not None:
            artifacts["shap_local"] = shap_local
            figures["shap_waterfall"] = plot_shap_waterfall(
                shap_local["values"],
                output_dir=self.config.get("output_dir", "reports"),
                instance_index=instance_index,
            )
        else:
            warnings.append(
                "SHAP local explanation was not generated. Install shap to enable "
                "local SHAP artifacts."
            )

        return ExplanationResult(
            model_type=type(self.classifier).__name__,
            scope="local",
            artifacts=artifacts,
            warnings=warnings,
            figures=figures,
            tables=tables,
        )

    def _native_feature_importance_frame(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return native tree feature importances as a ranked DataFrame."""

        return (
            pd.DataFrame(
                {
                    "feature": self._model_feature_names(x),
                    "importance": self.classifier.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


class TreeEnsembleExplainer(TreeShapMixin, ModelAgnosticExplainer):
    """Explain tree ensembles with native importance, permutation, and SHAP."""

    def explain_global(
        self,
        x: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> ExplanationResult:
        """Generate global explanations for a fitted tree ensemble."""

        native = self._native_feature_importance_frame(x)
        artifacts: dict[str, Any] = {
            **self._explanation_metadata(),
            "native_feature_importance": native,
        }
        warnings = self._common_warnings()
        figures: dict[str, str] = {
            "native_feature_importance": plot_native_feature_importance(
                native,
                output_dir=self.config.get("output_dir", "reports"),
                top_n=int(self.config.get("top_n_features", 15)),
            )
        }
        tables: dict[str, str] = {
            "native_feature_importance": self.save_table(
                native,
                "native_feature_importance.csv",
            )
        }

        if y is not None:
            permutation = self._permutation_importance_frame(x, y)
            artifacts["permutation_importance"] = permutation
            tables["permutation_importance"] = self.save_table(
                permutation,
                "permutation_importance.csv",
            )
            figures["permutation_importance"] = plot_permutation_importance(
                permutation,
                output_dir=self.config.get("output_dir", "reports"),
                top_n=int(self.config.get("top_n_features", 15)),
            )

        shap_result = self._shap_global(x)

        if shap_result is not None:
            artifacts["shap_global"] = shap_result
            figures["shap_global_bar"] = plot_shap_global_bar(
                shap_result["values"],
                output_dir=self.config.get("output_dir", "reports"),
            )
            figures["shap_beeswarm"] = plot_shap_beeswarm(
                shap_result["values"],
                output_dir=self.config.get("output_dir", "reports"),
            )
        else:
            warnings.append(
                "SHAP global explanation was not generated. "
                "Install shap to enable SHAP artifacts."
            )

        return ExplanationResult(
            model_type=type(self.classifier).__name__,
            scope="global",
            artifacts=artifacts,
            warnings=warnings,
            figures=figures,
            tables=tables,
        )

    def explain_local(self, x: pd.DataFrame, instance_index: int) -> ExplanationResult:
        """Generate a local model-agnostic explanation enriched with SHAP."""

        base_result = super().explain_local(x, instance_index)
        artifacts = dict(base_result.artifacts)
        warnings = list(base_result.warnings)

        shap_local = self._shap_local(x, instance_index)

        if shap_local is not None:
            artifacts["shap_local"] = shap_local
            figures = dict(base_result.figures or {})
            figures["shap_waterfall"] = plot_shap_waterfall(
                shap_local["values"],
                output_dir=self.config.get("output_dir", "reports"),
                instance_index=instance_index,
            )
        else:
            figures = dict(base_result.figures or {})
            warnings.append(
                "SHAP local explanation was not generated. "
                "Install shap to enable local SHAP artifacts."
            )

        return ExplanationResult(
            model_type=type(self.classifier).__name__,
            scope="local",
            artifacts=artifacts,
            warnings=warnings,
            figures=figures,
            tables=base_result.tables,
        )

    def _native_feature_importance_frame(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return native ensemble feature importances as a ranked DataFrame."""

        return (
            pd.DataFrame(
                {
                    "feature": self._model_feature_names(x),
                    "importance": self.classifier.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
