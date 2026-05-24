"""Model-agnostic explainers based on permutation importance and LIME."""

from typing import Any

import numpy as np
import pandas as pd

from explainability.base_explainer import BaseExplainer, ExplanationResult
from explainability.visualizations import (
    plot_lime_explanation,
    plot_permutation_importance,
)


class ModelAgnosticExplainer(BaseExplainer):
    """Explain arbitrary classifiers through model-agnostic techniques."""

    def explain_global(
        self,
        x: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> ExplanationResult:
        """Generate global permutation importance when target labels are available."""

        artifacts: dict[str, Any] = self._explanation_metadata()
        warnings = self._common_warnings()
        figures: dict[str, str] = {}
        tables: dict[str, str] = {}

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
        else:
            warnings.append("Global permutation importance requires y.")

        return ExplanationResult(
            model_type=type(self.classifier).__name__,
            scope="global",
            artifacts=artifacts,
            warnings=warnings,
            figures=figures,
            tables=tables,
        )

    def explain_local(self, x: pd.DataFrame, instance_index: int) -> ExplanationResult:
        """Generate a local LIME explanation and prediction summary."""

        x_df = self._as_dataframe(x)
        x_row = x_df.iloc[[instance_index]]

        artifacts: dict[str, Any] = {
            **self._explanation_metadata(),
            "prediction": self._predict_summary(x_row),
        }
        warnings = self._common_warnings() + self._consume_probability_warnings()
        figures: dict[str, str] = {}
        tables: dict[str, str] = {}

        lime_result = self._lime_explanation(x_df, instance_index)

        if lime_result is not None:
            artifacts["lime"] = lime_result
            if lime_result.get("html_path"):
                figures["lime"] = lime_result["html_path"]
            if lime_result.get("csv_path"):
                tables["lime"] = lime_result["csv_path"]
        else:
            warnings.append(
                "LIME explanation was not generated. "
                "Install lime and provide x_reference to enable local surrogate "
                "explanations."
            )
        warnings.extend(self._consume_probability_warnings())

        return ExplanationResult(
            model_type=type(self.classifier).__name__,
            scope="local",
            artifacts=artifacts,
            warnings=warnings,
            figures=figures,
            tables=tables,
        )

    def _lime_explanation(
        self, x: pd.DataFrame, instance_index: int
    ) -> dict[str, Any] | None:
        """Build and persist a LIME explanation for a single instance."""

        if self.x_reference is None:
            return None

        try:
            import lime.lime_tabular
        except ImportError:
            return None

        x_reference_df = self._as_dataframe(self.x_reference)
        feature_names = self._input_feature_names(x_reference_df)

        class_names = self.class_names

        if class_names is None and hasattr(self.classifier, "classes_"):
            class_names = [str(value) for value in self.classifier.classes_]

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.asarray(x_reference_df),
            feature_names=feature_names,
            class_names=class_names,
            mode="classification",
            discretize_continuous=bool(
                self.config.get("lime_discretize_continuous", True)
            ),
            random_state=self.config.get(
                "lime_random_state", self.config.get("random_state")
            ),
            categorical_features=self.config.get("categorical_features"),
            categorical_names=self.config.get("categorical_names"),
        )

        positive_class_index = self._configured_positive_class_index()
        if positive_class_index is None:
            return None

        labels = (positive_class_index,)

        explanation = explainer.explain_instance(
            data_row=np.asarray(x.iloc[instance_index]),
            predict_fn=self._predict_proba_like,
            num_features=int(self.config.get("lime_num_features", 10)),
            labels=labels,
        )
        paths = plot_lime_explanation(
            explanation,
            output_dir=self.config.get("output_dir", "reports"),
            instance_index=instance_index,
            label=labels[0],
        )
        lime_table = pd.DataFrame(
            explanation.as_list(label=labels[0]),
            columns=["feature", "weight"],
        )
        table_path = self.save_table(lime_table, f"lime_instance_{instance_index}.csv")

        return {
            "as_list": explanation.as_list(label=labels[0]),
            "explained_class": self._configured_positive_class(),
            "explained_class_index": labels[0],
            "score": explanation.score,
            "local_prediction": explanation.local_pred,
            "html_path": paths["html_path"],
            "csv_path": table_path,
            "figure_csv_path": paths["csv_path"],
            "used_real_predict_proba": self.has_predict_proba(),
        }
