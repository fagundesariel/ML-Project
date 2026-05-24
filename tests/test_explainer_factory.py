from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from explainability.explainers.knn_explainer import KNNExplainer
from explainability.explainers.linear_explainer import LinearModelExplainer
from explainability.explainers.svm_explainer import (
    LinearSVMExplainer,
    NonLinearSVMExplainer,
)
from explainability.explainers.tree_explainer import (
    DecisionTreeExplainer,
    TreeEnsembleExplainer,
)
from explainability.factory import ExplainerFactory
from modeling_utils import make_modeling_pipeline


def test_factory_selects_model_specific_explainers():
    cases = [
        (DecisionTreeClassifier(), DecisionTreeExplainer),
        (RandomForestClassifier(), TreeEnsembleExplainer),
        (KNeighborsClassifier(), KNNExplainer),
        (LogisticRegression(), LinearModelExplainer),
        (SVC(kernel="linear"), LinearSVMExplainer),
        (SVC(kernel="rbf"), NonLinearSVMExplainer),
    ]

    for classifier, expected in cases:
        pipeline = make_modeling_pipeline(classifier)
        explainer = ExplainerFactory.create(pipeline)
        assert isinstance(explainer, expected)
