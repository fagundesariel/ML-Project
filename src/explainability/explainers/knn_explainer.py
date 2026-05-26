"""KNN-specific explainer with nearest-neighbor local diagnostics."""

from typing import Any

import numpy as np
import pandas as pd

from explainability.base_explainer import ExplanationResult
from explainability.explainers.model_agnostic_explainer import ModelAgnosticExplainer


class KNNExplainer(ModelAgnosticExplainer):
    """Explain KNN predictions with model-agnostic outputs and neighbors."""

    def explain_local(self, x: pd.DataFrame, instance_index: int) -> ExplanationResult:
        """Generate a local explanation enriched with nearest-neighbor details."""

        base_result = super().explain_local(x, instance_index)
        x_df = self._as_dataframe(x)
        z = self._preprocess_for_classifier(x_df.iloc[[instance_index]])

        n_neighbors = int(self.config.get("n_neighbors", self.classifier.n_neighbors))
        n_samples_fit = getattr(self.classifier, "n_samples_fit_", n_neighbors)
        n_neighbors = min(n_neighbors, n_samples_fit)

        distances, indices = self.classifier.kneighbors(
            z,
            n_neighbors=n_neighbors,
            return_distance=True,
        )

        neighbor_frame = self._neighbor_frame(
            distances=distances[0],
            indices=indices[0],
        )

        artifacts = dict(base_result.artifacts)
        artifacts["neighbors"] = neighbor_frame
        tables = dict(base_result.tables or {})
        tables["neighbors"] = self.save_table(
            neighbor_frame,
            f"knn_neighbors_instance_{instance_index}.csv",
        )

        return ExplanationResult(
            model_type=type(self.classifier).__name__,
            scope="local",
            artifacts=artifacts,
            warnings=base_result.warnings,
            figures=base_result.figures,
            tables=tables,
        )

    def _neighbor_frame(
        self, distances: np.ndarray, indices: np.ndarray
    ) -> pd.DataFrame:
        """Build a table describing the fitted neighbors used by KNN."""

        rows: list[dict[str, Any]] = []

        fitted_targets = getattr(self.classifier, "_y", None)

        for rank, fitted_index in enumerate(indices, start=1):
            target = None

            if fitted_targets is not None:
                raw_target = np.asarray(fitted_targets)[fitted_index]
                target = self._decode_knn_target(raw_target)

            rows.append(
                {
                    "rank": rank,
                    "fitted_training_index": int(fitted_index),
                    "distance": float(distances[rank - 1]),
                    "neighbor_target": target,
                }
            )

        return pd.DataFrame(rows)

    def _decode_knn_target(self, raw_target: Any) -> Any:
        """Map encoded KNN target values back to classifier class labels."""

        if not hasattr(self.classifier, "classes_"):
            return raw_target

        classes = np.asarray(self.classifier.classes_)
        raw_array = np.asarray(raw_target)

        if raw_array.ndim == 0 and np.issubdtype(raw_array.dtype, np.integer):
            raw_index = int(raw_array)

            if 0 <= raw_index < len(classes):
                return classes[raw_index]

        return raw_target
