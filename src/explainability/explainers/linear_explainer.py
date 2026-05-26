"""Explainers for linear classifiers and linear decision functions."""

from typing import Any

import numpy as np
import pandas as pd

from explainability.base_explainer import BaseExplainer, ExplanationResult
from explainability.visualizations import (
    plot_linear_coefficients,
    plot_local_contributions,
    plot_permutation_importance,
)


class LinearModelExplainer(BaseExplainer):
    """Explain linear models with coefficients and local contributions."""

    def explain_global(
        self,
        x: pd.DataFrame,
        y: pd.Series | np.ndarray | None = None,
    ) -> ExplanationResult:
        """Generate coefficient-based global explanations and optional permutation."""

        coefficients = self._coefficient_frame(x)
        artifacts: dict[str, Any] = {
            **self._explanation_metadata(),
            "coefficients": coefficients,
        }
        warnings = self._common_warnings()
        figures: dict[str, str] = {
            "coefficients_abs": plot_linear_coefficients(
                coefficients,
                output_dir=self.config.get("output_dir", "reports"),
                top_n=int(self.config.get("top_n_features", 15)),
                signed=False,
            ),
            "coefficients_signed": plot_linear_coefficients(
                coefficients,
                output_dir=self.config.get("output_dir", "reports"),
                top_n=int(self.config.get("top_n_features", 15)),
                signed=True,
            ),
        }
        tables: dict[str, str] = {
            "coefficients": self.save_table(coefficients, "linear_coefficients.csv")
        }
        warnings.append(
            "Linear coefficients are interpreted on the preprocessed feature scale. "
            "If scaling or dimensionality reduction is used, coefficient magnitudes "
            "refer to transformed values rather than raw clinical units."
        )

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

        return ExplanationResult(
            model_type=type(self.classifier).__name__,
            scope="global",
            artifacts=artifacts,
            warnings=warnings,
            figures=figures,
            tables=tables,
        )

    def explain_local(self, x: pd.DataFrame, instance_index: int) -> ExplanationResult:
        """Generate local feature contributions for one instance."""

        x_df = self._as_dataframe(x)
        x_row = x_df.iloc[[instance_index]]
        z = self._preprocess_for_classifier(x_row)[0]

        coef = np.asarray(self.classifier.coef_)

        if coef.ndim == 1:
            coef = coef.reshape(1, -1)

        prediction = self.pipeline.predict(x_row)[0]
        coef_index = self._coef_index_for_prediction(prediction, coef)

        contributions = z * coef[coef_index]
        feature_names = self._model_feature_names(x_df)

        contribution_frame = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "value_after_preprocessing": z,
                    "coefficient": coef[coef_index],
                    "contribution": contributions,
                    "abs_contribution": np.abs(contributions),
                }
            )
            .sort_values("abs_contribution", ascending=False)
            .reset_index(drop=True)
        )

        artifacts = {
            **self._explanation_metadata(),
            "prediction": self._predict_summary(x_row),
            "local_contributions": contribution_frame,
        }
        table_path = self.save_table(
            contribution_frame,
            f"local_contributions_instance_{instance_index}.csv",
        )
        figure_path = plot_local_contributions(
            contribution_frame,
            output_dir=self.config.get("output_dir", "reports"),
            instance_index=instance_index,
            top_n=int(self.config.get("top_n_features", 15)),
        )

        return ExplanationResult(
            model_type=type(self.classifier).__name__,
            scope="local",
            artifacts=artifacts,
            warnings=self._common_warnings() + self._consume_probability_warnings(),
            figures={"local_contributions": figure_path},
            tables={"local_contributions": table_path},
        )

    def _coefficient_frame(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return model coefficients as a long-form ranked DataFrame."""

        coef = np.asarray(self.classifier.coef_)

        if coef.ndim == 1:
            coef = coef.reshape(1, -1)

        feature_names = self._model_feature_names(x)

        rows: list[dict[str, Any]] = []

        for class_index, class_coef in enumerate(coef):
            class_label = self._class_label_for_coef_index(class_index, coef)

            for feature, coefficient in zip(feature_names, class_coef):
                rows.append(
                    {
                        "class": class_label,
                        "feature": feature,
                        "coefficient": coefficient,
                        "abs_coefficient": abs(coefficient),
                    }
                )

        return (
            pd.DataFrame(rows)
            .sort_values(["class", "abs_coefficient"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def _coef_index_for_prediction(self, prediction: Any, coef: np.ndarray) -> int:
        """Select the coefficient row associated with a predicted class."""

        if coef.shape[0] == 1:
            return 0

        positive_class = self.config.get("positive_class")
        if positive_class is not None and hasattr(self.classifier, "classes_"):
            matches = np.where(self.classifier.classes_ == positive_class)[0]
            if len(matches) > 0:
                return int(matches[0])

        if hasattr(self.classifier, "classes_"):
            matches = np.where(self.classifier.classes_ == prediction)[0]

            if len(matches) > 0:
                return int(matches[0])

        return 0

    def _class_label_for_coef_index(self, class_index: int, coef: np.ndarray) -> str:
        """Return the class label represented by a coefficient row."""

        if hasattr(self.classifier, "classes_"):
            classes = list(self.classifier.classes_)

            if coef.shape[0] == 1 and len(classes) == 2:
                return str(classes[1])

            if class_index < len(classes):
                return str(classes[class_index])

        return f"class_{class_index}"
