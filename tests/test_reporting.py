import pandas as pd

from explainability.base_explainer import ExplanationResult
from explainability.reporting import (
    consolidate_feature_rankings,
    save_explainability_report,
)


def test_consolidate_feature_rankings_writes_top_features(tmp_path):
    result = ExplanationResult(
        model_type="RandomForestClassifier",
        scope="global",
        artifacts={
            "permutation_importance": pd.DataFrame(
                {
                    "feature": ["a", "b"],
                    "importance_mean": [0.2, 0.1],
                    "importance_std": [0.01, 0.02],
                }
            ),
            "native_feature_importance": pd.DataFrame(
                {"feature": ["b", "a"], "importance": [0.8, 0.2]}
            ),
        },
        warnings=[],
    )

    table = consolidate_feature_rankings(result, top_n=2, output_dir=tmp_path)

    assert (tmp_path / "tables" / "explainability_top_features.csv").exists()
    assert "consensus_score" in table.columns


def test_report_markdown_is_created(tmp_path):
    result = ExplanationResult(
        model_type="LogisticRegression",
        scope="global",
        artifacts={"explained_class": "ckd"},
        warnings=["No causal interpretation."],
    )
    top_features = pd.DataFrame(
        {
            "feature": ["a"],
            "consensus_score": [1.0],
            "interpretation_hint": [
                "Higher model reliance according to permutation importance"
            ],
        }
    )

    path = save_explainability_report(
        model_name="LogisticRegression",
        global_result=result,
        local_results=[],
        top_features=top_features,
        output_dir=tmp_path,
    )

    assert (tmp_path / "explainability_summary.md").exists()
    report = open(path, encoding="utf-8").read()
    assert "# Explainability Analysis" in report
    assert "Explained class: ckd" in report
    assert "Positive SHAP values indicate increased support for this class." in report
