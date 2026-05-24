"""SVM explainers for linear and kernel-based classifiers."""

from explainability.explainers.linear_explainer import LinearModelExplainer
from explainability.explainers.model_agnostic_explainer import ModelAgnosticExplainer


class LinearSVMExplainer(LinearModelExplainer):
    """Explain linear SVM-style classifiers through their coefficients."""

    pass


class NonLinearSVMExplainer(ModelAgnosticExplainer):
    """Explain nonlinear SVM classifiers with model-agnostic methods."""

    def explain_local(self, x, instance_index: int):
        """Generate a local explanation with SVM probability calibration warnings."""

        result = super().explain_local(x, instance_index)
        warnings = list(result.warnings)
        if not self.has_predict_proba():
            warnings.append(
                "LIME explanations for SVM without calibrated probabilities rely on "
                "approximated probabilities from decision_function."
            )
            warnings.append(
                "For SVC(kernel='rbf'), prefer probability=True or "
                "CalibratedClassifierCV when probability-based local explanations "
                "are required."
            )
        return type(result)(
            model_type=result.model_type,
            scope=result.scope,
            artifacts=result.artifacts,
            warnings=warnings,
            figures=result.figures,
            tables=result.tables,
            report_text=result.report_text,
        )
