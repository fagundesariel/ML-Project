import pandas as pd
from imblearn import FunctionSampler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from explainability.explainers.model_agnostic_explainer import ModelAgnosticExplainer
from modeling_utils import make_modeling_pipeline


def _data():
    x = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1, 0],
            "c": [0, 0, 1, 1, 1, 0],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])
    return x, y


def test_permutation_importance_frame_has_expected_columns(tmp_path):
    x, y = _data()
    pipeline = make_modeling_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(x, y)
    explainer = ModelAgnosticExplainer(
        pipeline,
        feature_names=x.columns,
        config={"output_dir": tmp_path, "permutation_n_repeats": 2},
    )

    frame = explainer._permutation_importance_frame(x, y)

    assert list(frame.columns) == ["feature", "importance_mean", "importance_std"]
    assert set(frame["feature"]) == set(x.columns)


def test_reducer_generates_warning():
    x, y = _data()
    pipeline = make_modeling_pipeline(
        LogisticRegression(max_iter=200), reducer=PCA(n_components=2)
    )
    pipeline.fit(x, y)
    explainer = ModelAgnosticExplainer(pipeline, feature_names=x.columns)

    warnings = explainer._common_warnings()

    assert any("dimensionality reduction" in warning for warning in warnings)


def test_sampler_is_ignored_during_explain_preprocess():
    x, y = _data()
    pipeline = make_modeling_pipeline(
        LogisticRegression(max_iter=200),
        sampler=FunctionSampler(),
    )
    pipeline.fit(x, y)
    explainer = ModelAgnosticExplainer(pipeline, feature_names=x.columns)

    transformed = explainer._preprocess_for_classifier(x)

    assert transformed.shape == x.shape


def test_positive_class_is_not_inferred_from_second_classifier_class():
    x = pd.DataFrame({"a": [0, 1, 2, 3], "b": [3, 2, 1, 0]})
    y = pd.Series(["ckd", "ckd", "notckd", "notckd"])
    pipeline = make_modeling_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(x, y)
    explainer = ModelAgnosticExplainer(pipeline, feature_names=x.columns)

    assert list(pipeline.named_steps["classifier"].classes_) == ["ckd", "notckd"]
    assert explainer._configured_positive_class() is None
    assert explainer._configured_positive_class_index() is None


def test_configured_positive_class_uses_matching_classifier_index():
    x = pd.DataFrame({"a": [0, 1, 2, 3], "b": [3, 2, 1, 0]})
    y = pd.Series(["ckd", "ckd", "notckd", "notckd"])
    pipeline = make_modeling_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(x, y)
    explainer = ModelAgnosticExplainer(
        pipeline,
        feature_names=x.columns,
        config={"positive_class": "ckd"},
    )

    assert explainer._configured_positive_class() == "ckd"
    assert explainer._configured_positive_class_index() == 0
