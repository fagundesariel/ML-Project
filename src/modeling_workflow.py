"""Workflow wrapper that combines model fitting with explainability artifacts."""

from explainability.factory import ExplainerFactory
from explainability.run_explainability import generate_explainability_artifacts
from modeling_utils import make_modeling_pipeline


class ModelingWorkflow:
    """Coordinate a fitted pipeline and optional explainability operations."""

    def __init__(
        self,
        pipeline,
        *,
        enable_explainability=False,
        feature_names=None,
        class_names=None,
        explainability_config=None,
    ):
        """Store the modeling pipeline and explainability configuration."""

        self.pipeline = pipeline
        self.enable_explainability = enable_explainability
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainability_config = explainability_config or {}
        self.explainer = None
        self.x_reference = None
        self.y_reference = None

    def fit(self, X, y):
        """Fit the pipeline and initialize an explainer when enabled."""

        self.pipeline.fit(X, y)
        self.x_reference = X
        self.y_reference = y

        if self.class_names is None and hasattr(
            self.pipeline.named_steps["classifier"],
            "classes_",
        ):
            self.class_names = [
                str(value) for value in self.pipeline.named_steps["classifier"].classes_
            ]

        if self.enable_explainability:
            self.explainer = ExplainerFactory.create(
                pipeline=self.pipeline,
                feature_names=self.feature_names,
                class_names=self.class_names,
                x_reference=X,
                y_reference=y,
                config=self.explainability_config,
            )

        return self

    def predict(self, X):
        """Predict class labels with the wrapped pipeline."""

        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities when the wrapped classifier supports them."""

        classifier = self.pipeline.named_steps["classifier"]
        if not hasattr(classifier, "predict_proba"):
            raise AttributeError(
                "The fitted pipeline does not expose real predict_proba. "
                "Approximate probabilities are available only inside explainability "
                "helpers."
            )
        return self.pipeline.predict_proba(X)

    def explain_global(self, X, y=None):
        """Generate global explanations for the fitted workflow."""

        if not self.enable_explainability or self.explainer is None:
            raise RuntimeError("Explainability is not enabled for this workflow.")

        return self.explainer.explain_global(X, y)

    def explain_local(self, X, instance_index: int):
        """Generate a local explanation for one row of input data."""

        if not self.enable_explainability or self.explainer is None:
            raise RuntimeError("Explainability is not enabled for this workflow.")

        return self.explainer.explain_local(X, instance_index)

    def generate_explainability_artifacts(
        self,
        X,
        y=None,
        *,
        instance_indices: list[int] | None = None,
        output_dir="reports",
    ):
        """Create report, table, and figure artifacts for explanations."""

        if not self.enable_explainability or self.explainer is None:
            raise RuntimeError("Explainability is not enabled for this workflow.")

        return generate_explainability_artifacts(
            self.explainer,
            X,
            y,
            instance_indices=instance_indices,
            output_dir=output_dir,
        )


def make_modeling_workflow(
    classifier,
    *,
    sampler=None,
    reducer=None,
    scaler=None,
    enable_explainability=False,
    feature_names=None,
    class_names=None,
    explainability_config=None,
):
    """Build a modeling pipeline and wrap it in a workflow object."""

    pipeline = make_modeling_pipeline(
        classifier=classifier,
        sampler=sampler,
        reducer=reducer,
        scaler=scaler,
    )

    return ModelingWorkflow(
        pipeline=pipeline,
        enable_explainability=enable_explainability,
        feature_names=feature_names,
        class_names=class_names,
        explainability_config=explainability_config,
    )
