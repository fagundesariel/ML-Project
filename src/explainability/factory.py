"""Factory for selecting the best explainer for a fitted classifier."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from explainability.base_explainer import BaseExplainer
from explainability.explainers.knn_explainer import KNNExplainer
from explainability.explainers.linear_explainer import LinearModelExplainer
from explainability.explainers.model_agnostic_explainer import ModelAgnosticExplainer
from explainability.explainers.svm_explainer import (
    LinearSVMExplainer,
    NonLinearSVMExplainer,
)
from explainability.explainers.tree_explainer import (
    DecisionTreeExplainer,
    TreeEnsembleExplainer,
)


class ExplainerFactory:
    """Create model-specific explainers from sklearn-compatible pipelines."""

    @staticmethod
    def create(
        pipeline,
        *,
        feature_names: list[str] | pd.Index | None = None,
        class_names: list[str] | None = None,
        x_reference: pd.DataFrame | np.ndarray | None = None,
        y_reference: pd.Series | np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> BaseExplainer:
        """Return an explainer implementation matching the pipeline classifier."""

        classifier = pipeline.named_steps["classifier"]

        if isinstance(classifier, DecisionTreeClassifier):
            return DecisionTreeExplainer(
                pipeline,
                feature_names=feature_names,
                class_names=class_names,
                x_reference=x_reference,
                y_reference=y_reference,
                config=config,
            )

        if isinstance(classifier, RandomForestClassifier):
            return TreeEnsembleExplainer(
                pipeline,
                feature_names=feature_names,
                class_names=class_names,
                x_reference=x_reference,
                y_reference=y_reference,
                config=config,
            )

        if isinstance(classifier, KNeighborsClassifier):
            return KNNExplainer(
                pipeline,
                feature_names=feature_names,
                class_names=class_names,
                x_reference=x_reference,
                y_reference=y_reference,
                config=config,
            )

        if isinstance(classifier, LogisticRegression):
            return LinearModelExplainer(
                pipeline,
                feature_names=feature_names,
                class_names=class_names,
                x_reference=x_reference,
                y_reference=y_reference,
                config=config,
            )

        if isinstance(classifier, (LinearSVC, RidgeClassifier)):
            return LinearSVMExplainer(
                pipeline,
                feature_names=feature_names,
                class_names=class_names,
                x_reference=x_reference,
                y_reference=y_reference,
                config=config,
            )

        if isinstance(classifier, SGDClassifier):
            if getattr(classifier, "loss", None) in {
                "hinge",
                "log_loss",
                "modified_huber",
            }:
                return LinearModelExplainer(
                    pipeline,
                    feature_names=feature_names,
                    class_names=class_names,
                    x_reference=x_reference,
                    y_reference=y_reference,
                    config=config,
                )

        if isinstance(classifier, SVC):
            if classifier.kernel == "linear":
                return LinearSVMExplainer(
                    pipeline,
                    feature_names=feature_names,
                    class_names=class_names,
                    x_reference=x_reference,
                    y_reference=y_reference,
                    config=config,
                )

            return NonLinearSVMExplainer(
                pipeline,
                feature_names=feature_names,
                class_names=class_names,
                x_reference=x_reference,
                y_reference=y_reference,
                config=config,
            )

        return ModelAgnosticExplainer(
            pipeline,
            feature_names=feature_names,
            class_names=class_names,
            x_reference=x_reference,
            y_reference=y_reference,
            config=config,
        )
