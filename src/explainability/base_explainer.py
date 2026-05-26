"""Base classes and shared utilities for model explainers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from modeling_utils import DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class ExplanationResult:
    """Container for explanation outputs and generated artifact paths."""

    model_type: str
    scope: str
    artifacts: dict[str, Any]
    warnings: list[str]
    figures: dict[str, str] | None = None
    tables: dict[str, str] | None = None
    report_text: str | None = None


class BaseExplainer(ABC):
    """Define the common interface and helpers for all explainers."""

    def __init__(
        self,
        pipeline,
        *,
        feature_names: list[str] | pd.Index | None = None,
        class_names: list[str] | None = None,
        x_reference: pd.DataFrame | np.ndarray | None = None,
        y_reference: pd.Series | np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize an explainer for a fitted sklearn-compatible pipeline."""

        self.pipeline = pipeline
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.class_names = class_names
        self.x_reference = x_reference
        self.y_reference = y_reference
        self.config = config or {}
        self._probability_warnings: list[str] = []

    @abstractmethod
    def explain_global(
        self, x: pd.DataFrame, y: pd.Series | np.ndarray | None = None
    ) -> ExplanationResult:
        """Return model-level explanation artifacts for a dataset."""

        raise NotImplementedError

    @abstractmethod
    def explain_local(self, x: pd.DataFrame, instance_index: int) -> ExplanationResult:
        """Return instance-level explanation artifacts for one observation."""

        raise NotImplementedError

    @property
    def classifier(self):
        """Return the classifier step from the wrapped pipeline."""

        return self.pipeline.named_steps["classifier"]

    def _as_dataframe(self, x: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return input data as a DataFrame with known feature names when possible."""

        if isinstance(x, pd.DataFrame):
            return x.copy()

        if self.feature_names is not None:
            return pd.DataFrame(x, columns=self.feature_names)

        return pd.DataFrame(x)

    def _preprocess_for_classifier(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Apply only transformer-like steps before the classifier.

        Samplers are intentionally skipped here because they are applied only
        during fit, not during predict/explain.
        """
        data = self._as_dataframe(x)

        for step_name, step in self.pipeline.steps:
            if step_name == "classifier":
                break

            if hasattr(step, "transform"):
                data = step.transform(data)
                continue

            # imblearn samplers usually implement fit_resample, not transform.
            if hasattr(step, "fit_resample"):
                continue

        return np.asarray(data)

    def _model_feature_names(self, x: pd.DataFrame | np.ndarray) -> list[str]:
        """Return feature names after preprocessing steps that change dimensions."""

        names = self.feature_names

        if names is None:
            x_df = self._as_dataframe(x)
            names = list(x_df.columns)

        current_names = list(names)

        for step_name, step in self.pipeline.steps:
            if step_name == "classifier":
                break

            if hasattr(step, "get_feature_names_out"):
                try:
                    current_names = list(step.get_feature_names_out(current_names))
                    continue
                except Exception:
                    pass

            if step_name == "reducer":
                transformed = self._preprocess_for_classifier(x)
                return [f"component_{idx + 1}" for idx in range(transformed.shape[1])]

        transformed = self._preprocess_for_classifier(x)

        if len(current_names) != transformed.shape[1]:
            return [f"feature_{idx + 1}" for idx in range(transformed.shape[1])]

        return current_names

    def _input_feature_names(self, x: pd.DataFrame | np.ndarray) -> list[str]:
        """Return feature names in the original model input space."""

        if self.feature_names is not None:
            return list(self.feature_names)

        x_df = self._as_dataframe(x)
        return list(x_df.columns)

    def _predict_summary(self, x_row: pd.DataFrame) -> dict[str, Any]:
        """Return prediction and optional probability details for one row."""

        prediction = self.pipeline.predict(x_row)[0]

        summary: dict[str, Any] = {
            "prediction": prediction,
        }

        try:
            probabilities = self._predict_proba_like(x_row)[0]
            summary["probabilities"] = probabilities
        except Exception:
            summary["probabilities"] = None

        return summary

    def _predict_proba_like(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return real or approximated class probabilities for local explainers."""

        x_df = self._as_dataframe(x)

        if self.has_predict_proba():
            return self.pipeline.predict_proba(x_df)

        if hasattr(self.pipeline, "decision_function"):
            self._add_probability_warning(
                "Probabilities were approximated from decision_function and may not "
                "be calibrated."
            )
            scores = self.pipeline.decision_function(x_df)
            scores = np.asarray(scores)

            if scores.ndim == 1:
                positive = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1.0 - positive, positive])

            shifted = scores - scores.max(axis=1, keepdims=True)
            exp_scores = np.exp(shifted)
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)

        raise AttributeError(
            "The pipeline does not expose predict_proba or decision_function."
        )

    def _add_probability_warning(self, warning: str) -> None:
        """Record a probability warning once per explainer instance."""

        if warning not in self._probability_warnings:
            self._probability_warnings.append(warning)

    def _consume_probability_warnings(self) -> list[str]:
        """Return pending probability warnings and clear the buffer."""

        warnings = list(self._probability_warnings)
        self._probability_warnings.clear()
        return warnings

    def _permutation_importance_frame(
        self,
        x: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """Compute permutation importance and return it as a ranked DataFrame."""

        n_repeats = int(self.config.get("permutation_n_repeats", 10))
        random_state = int(self.config.get("random_state", DEFAULT_RANDOM_STATE))
        scoring = self.config.get("scoring", "f1_macro")
        n_jobs = self.config.get("n_jobs", 1)

        result = permutation_importance(
            self.pipeline,
            x,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=n_jobs,
        )

        return (
            pd.DataFrame(
                {
                    "feature": self._input_feature_names(x),
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

    def _common_warnings(self) -> list[str]:
        """Return warnings shared by explainers for the current pipeline."""

        warnings: list[str] = []

        if self.has_dimensionality_reduction():
            warnings.append(
                "The pipeline contains a dimensionality reduction step. "
                "Native model explanations may refer to transformed components, "
                "not directly to original clinical variables."
            )

        return warnings

    def _configured_positive_class(self) -> Any:
        """Return the class configured for class-specific explanations."""

        if "positive_class" in self.config:
            return self.config["positive_class"]

        return None

    def _configured_positive_class_index(self) -> int | None:
        """Return the classifier index for the configured positive class."""

        selected_class = self._configured_positive_class()
        classes = list(getattr(self.classifier, "classes_", []))
        if selected_class is None or not classes:
            return None

        matches = np.where(np.asarray(classes) == selected_class)[0]
        if len(matches) > 0:
            return int(matches[0])

        raise ValueError(
            f"Configured positive_class={selected_class!r} is not present in "
            f"classifier.classes_: {classes}. Review the target mapping before "
            "generating class-specific explanations."
        )

    def _explanation_metadata(self) -> dict[str, Any]:
        """Return metadata that documents the explained class."""

        classes = list(getattr(self.classifier, "classes_", []))
        return {
            "class_names": self.class_names,
            "classifier_classes": classes,
            "explained_class": self._configured_positive_class(),
        }

    def has_dimensionality_reduction(self) -> bool:
        """Return whether the pipeline contains a known dimension reducer."""

        reducer_names = {"reducer", "pca", "svd", "dimensionality_reduction"}
        for step_name, step in self.pipeline.steps:
            if step_name in reducer_names:
                return True
            module_name = type(step).__module__.lower()
            class_name = type(step).__name__.lower()
            if "decomposition" in module_name or class_name in {"pca", "truncatedsvd"}:
                return True
        return False

    def has_predict_proba(self) -> bool:
        """Return whether both classifier and pipeline expose probabilities."""

        return hasattr(self.classifier, "predict_proba") and hasattr(
            self.pipeline, "predict_proba"
        )

    def get_output_dir(self, *parts: str, create: bool = True) -> Path:
        """Return an output directory below the configured reports path."""

        base = Path(self.config.get("output_dir", "reports"))
        path = base.joinpath(*parts)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def save_table(self, frame: pd.DataFrame, filename: str) -> str:
        """Save a DataFrame artifact and return its filesystem path."""

        table_dir = self.get_output_dir("tables")
        path = table_dir / filename
        frame.to_csv(path, index=False)
        return str(path)
