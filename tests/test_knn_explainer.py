import importlib.util

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from explainability.explainers.knn_explainer import KNNExplainer
from modeling_utils import make_modeling_pipeline


def test_knn_local_explanation_includes_neighbors_and_lime_when_available(tmp_path):
    x = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1, 0],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])
    pipeline = make_modeling_pipeline(KNeighborsClassifier(n_neighbors=3))
    pipeline.fit(x, y)
    explainer = KNNExplainer(
        pipeline,
        feature_names=x.columns,
        x_reference=x,
        y_reference=y,
        config={"output_dir": tmp_path, "lime_num_features": 2, "positive_class": 1},
    )

    result = explainer.explain_local(x, 0)

    assert "neighbors" in result.artifacts
    assert {"rank", "fitted_training_index", "distance", "neighbor_target"}.issubset(
        result.artifacts["neighbors"].columns
    )
    if importlib.util.find_spec("lime") is not None:
        assert "lime" in result.artifacts
