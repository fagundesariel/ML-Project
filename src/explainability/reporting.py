"""Utilities for consolidating explanation outputs into tables and reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from explainability.base_explainer import ExplanationResult


def _table_dir(output_dir: str | Path = "reports") -> Path:
    path = Path(output_dir) / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rank(frame: pd.DataFrame, value_col: str, rank_col: str) -> pd.DataFrame:
    data = frame.copy()
    data[rank_col] = data[value_col].rank(method="min", ascending=False)
    return data


def _artifact_frame(result: ExplanationResult, *names: str) -> pd.DataFrame | None:
    for name in names:
        value = result.artifacts.get(name)
        if isinstance(value, pd.DataFrame):
            return value.copy()
    return None


def build_top_features_table(
    rankings: pd.DataFrame,
    *,
    top_n: int = 10,
    output_dir: str | Path = "reports",
) -> pd.DataFrame:
    """Save and return the top-ranked features from a consolidated ranking."""

    table = rankings.head(top_n).copy()
    path = _table_dir(output_dir) / "explainability_top_features.csv"
    table.to_csv(path, index=False)
    return table


def consolidate_feature_rankings(
    global_result: ExplanationResult,
    top_n: int = 10,
    *,
    output_dir: str | Path = "reports",
) -> pd.DataFrame:
    """Merge available explanation rankings into one consensus feature table."""

    frames: list[pd.DataFrame] = []

    permutation = _artifact_frame(global_result, "permutation_importance")
    if permutation is not None:
        permutation.to_csv(
            _table_dir(output_dir) / "permutation_importance.csv", index=False
        )
        frames.append(
            _rank(permutation, "importance_mean", "rank_permutation")[
                ["feature", "importance_mean", "importance_std", "rank_permutation"]
            ].rename(
                columns={
                    "importance_mean": "permutation_importance_mean",
                    "importance_std": "permutation_importance_std",
                }
            )
        )

    native = _artifact_frame(
        global_result,
        "native_feature_importance",
        "feature_importance",
    )
    if native is not None:
        native.to_csv(
            _table_dir(output_dir) / "native_feature_importance.csv",
            index=False,
        )
        frames.append(
            _rank(native, "importance", "rank_native")[
                ["feature", "importance", "rank_native"]
            ].rename(columns={"importance": "native_importance"})
        )

    coefficients = _artifact_frame(global_result, "coefficients")
    if coefficients is not None:
        coefficients.to_csv(
            _table_dir(output_dir) / "linear_coefficients.csv", index=False
        )
        coef = (
            coefficients.groupby("feature", as_index=False)
            .agg({"abs_coefficient": "max"})
            .sort_values("abs_coefficient", ascending=False)
        )
        frames.append(
            _rank(coef, "abs_coefficient", "rank_coefficient")[
                ["feature", "abs_coefficient", "rank_coefficient"]
            ]
        )

    shap_global = global_result.artifacts.get("shap_global")
    shap_frame = None
    if isinstance(shap_global, dict):
        shap_frame = shap_global.get("mean_abs_shap")
    if isinstance(shap_frame, pd.DataFrame):
        shap_frame.to_csv(
            _table_dir(output_dir) / "shap_global_importance.csv", index=False
        )
        frames.append(
            _rank(shap_frame, "mean_abs_shap", "rank_shap")[
                ["feature", "mean_abs_shap", "rank_shap"]
            ]
        )

    if not frames:
        empty = pd.DataFrame(
            columns=[
                "feature",
                "permutation_importance_mean",
                "permutation_importance_std",
                "native_importance",
                "abs_coefficient",
                "mean_abs_shap",
                "rank_permutation",
                "rank_native",
                "rank_coefficient",
                "rank_shap",
                "consensus_score",
                "interpretation_hint",
            ]
        )
        empty.to_csv(
            _table_dir(output_dir) / "explainability_top_features.csv", index=False
        )
        return empty

    consolidated = frames[0]
    for frame in frames[1:]:
        consolidated = consolidated.merge(frame, on="feature", how="outer")

    rank_cols = [
        "rank_permutation",
        "rank_native",
        "rank_coefficient",
        "rank_shap",
    ]
    present_rank_cols = [col for col in rank_cols if col in consolidated.columns]
    max_rank = max(len(consolidated), 1)
    scores = []
    for _, row in consolidated.iterrows():
        score = 0.0
        for col in present_rank_cols:
            if pd.notna(row.get(col)):
                score += (max_rank - float(row[col]) + 1.0) / max_rank
        scores.append(score)
    consolidated["consensus_score"] = scores

    for col in [
        "permutation_importance_mean",
        "permutation_importance_std",
        "native_importance",
        "abs_coefficient",
        "mean_abs_shap",
        *rank_cols,
    ]:
        if col not in consolidated.columns:
            consolidated[col] = np.nan

    consolidated["interpretation_hint"] = consolidated.apply(
        _interpretation_hint,
        axis=1,
    )
    consolidated = consolidated[
        [
            "feature",
            "permutation_importance_mean",
            "permutation_importance_std",
            "native_importance",
            "abs_coefficient",
            "mean_abs_shap",
            "rank_permutation",
            "rank_native",
            "rank_coefficient",
            "rank_shap",
            "consensus_score",
            "interpretation_hint",
        ]
    ].sort_values("consensus_score", ascending=False)

    return build_top_features_table(
        consolidated.reset_index(drop=True),
        top_n=top_n,
        output_dir=output_dir,
    )


def _interpretation_hint(row: pd.Series) -> str:
    hints: list[str] = []
    if pd.notna(row.get("rank_permutation")):
        hints.append("Higher model reliance according to permutation importance")
    if pd.notna(row.get("rank_shap")):
        hints.append("High average absolute SHAP contribution")
    if pd.notna(row.get("rank_coefficient")):
        hints.append("Large absolute linear coefficient")
    if pd.notna(row.get("rank_native")):
        hints.append("High model-specific feature importance")
    return "; ".join(hints) if hints else "Feature identified by available ranking"


def generate_global_interpretation_text(
    global_result: ExplanationResult,
    top_features: pd.DataFrame,
) -> str:
    """Create a concise narrative summary of global explanation results."""

    feature_list = ", ".join(top_features["feature"].astype(str).head(5))
    methods = []
    if "permutation_importance" in global_result.artifacts:
        methods.append("permutation importance")
    if any(
        name in global_result.artifacts
        for name in ("native_feature_importance", "feature_importance", "coefficients")
    ):
        methods.append("model-specific importance")
    if "shap_global" in global_result.artifacts:
        methods.append("SHAP")
    methods_text = ", ".join(methods) if methods else "the available explainer outputs"
    explained_class = _explained_class(global_result)
    shap_text = ""
    if explained_class is not None and "shap_global" in global_result.artifacts:
        shap_text = (
            f" SHAP values were computed for the configured explained class "
            f"({explained_class}); positive SHAP values indicate increased support "
            "for this class and negative values indicate reduced support."
        )
    return (
        f"Global interpretability for {global_result.model_type} used {methods_text}. "
        "The highest-ranked features associated with model predictions were: "
        f"{feature_list}.{shap_text}"
    )


def generate_local_interpretation_text(
    local_results: list[ExplanationResult],
    *,
    instance_indices: list[int],
) -> str:
    """Create a concise narrative summary of local explanation results."""

    if not local_results:
        return "No local explanation was generated."
    descriptions = []
    for idx, result in zip(instance_indices, local_results):
        prediction = result.artifacts.get("prediction", {}).get("prediction")
        available = [
            name
            for name in (
                "shap_local",
                "lime",
                "decision_path",
                "neighbors",
                "local_contributions",
            )
            if name in result.artifacts
        ]
        descriptions.append(
            f"Instance {idx} was predicted as {prediction}; local artifacts include "
            f"{', '.join(available) if available else 'prediction summary'}."
        )
    return " ".join(descriptions)


def generate_paper_ready_explainability_section(
    *,
    model_name: str,
    top_features: pd.DataFrame,
    figures: dict[str, str] | None = None,
    local_instance_index: int | None = None,
    warnings: list[str] | None = None,
    explained_class: object | None = None,
) -> str:
    """Generate a paragraph suitable for a methods or results section."""

    features = ", ".join(top_features["feature"].astype(str).head(5))
    figures_text = ", ".join(sorted((figures or {}).keys())) or "no figures"
    local_text = (
        f" Local explanations were generated for instance {local_instance_index}."
        if local_instance_index is not None
        else ""
    )
    warning_text = (
        " Methodological warnings included: " + " ".join(warnings) if warnings else ""
    )
    shap_class_text = (
        f" SHAP values were interpreted for the configured explained class "
        f"({explained_class}); positive SHAP values indicate increased support for "
        "this class and negative SHAP values indicate reduced support."
        if explained_class is not None
        else ""
    )
    return (
        f"To interpret the selected CKD prediction model ({model_name}), we combined "
        "global and local explainability techniques. Global explanations were obtained "
        "using permutation feature importance and, when applicable, model-specific "
        "feature importance. For tree-based models, class-specific SHAP values were "
        "also computed for the configured explained class when available. Local "
        "explanations were generated for representative test instances using SHAP "
        "waterfall plots and LIME when available. The most relevant predictors "
        "associated with model "
        f"predictions were {features}. Generated explainability figures included "
        f"{figures_text}.{shap_class_text}{local_text} "
        "The analysis should be interpreted as "
        f"model-based predictive evidence rather than causal inference.{warning_text}"
    )


def save_explainability_report(
    *,
    model_name: str,
    global_result: ExplanationResult,
    local_results: list[ExplanationResult],
    top_features: pd.DataFrame,
    figures: dict[str, str] | None = None,
    tables: dict[str, str] | None = None,
    instance_indices: list[int] | None = None,
    output_dir: str | Path = "reports",
) -> str:
    """Write a Markdown explainability report and return its path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    instance_indices = instance_indices or []
    warnings = _unique_warnings([global_result, *local_results])
    explained_class = _explained_class(global_result)
    paper_text = generate_paper_ready_explainability_section(
        model_name=model_name,
        top_features=top_features,
        figures=figures,
        local_instance_index=instance_indices[0] if instance_indices else None,
        warnings=warnings,
        explained_class=explained_class,
    )
    explained_class_text = (
        f"Explained class: {explained_class}\n\n"
        "Positive SHAP values indicate increased support for this class. "
        "Negative SHAP values indicate reduced support for this class."
        if explained_class is not None
        else "Explained class: not available from the explanation artifacts."
    )
    report = "\n\n".join(
        [
            "# Explainability Analysis",
            "## Model\n" + model_name,
            "## Explained class\n" + explained_class_text,
            "## Global interpretability\n"
            + generate_global_interpretation_text(global_result, top_features),
            "## Local interpretability\n"
            + generate_local_interpretation_text(
                local_results,
                instance_indices=instance_indices,
            ),
            "## Most relevant features for CKD prediction\n"
            + _markdown_feature_list(top_features),
            "## Agreement between methods\n"
            + "Consensus scores combine the available rankings and favor features that "
            "appear near the top in more than one explainability method.",
            "## Limitations\n"
            + "These explanations describe predictive patterns learned by the model "
            "and must not be interpreted as causal effects. If a dimensionality "
            "reducer such as PCA is present, native model explanations may refer "
            "to transformed "
            "components rather than directly to clinical variables. The analysis also "
            "depends on a leakage-free train/test split and preprocessing pipeline.",
            "## Paper-ready paragraph\n" + paper_text,
        ]
    )
    path = output_path / "explainability_summary.md"
    path.write_text(report, encoding="utf-8")
    return str(path)


def _unique_warnings(results: list[ExplanationResult]) -> list[str]:
    warnings: list[str] = []
    for result in results:
        for warning in result.warnings:
            if warning not in warnings:
                warnings.append(warning)
    return warnings


def _explained_class(result: ExplanationResult) -> object | None:
    shap_global = result.artifacts.get("shap_global")
    if isinstance(shap_global, dict) and "explained_class" in shap_global:
        return shap_global["explained_class"]

    if "explained_class" in result.artifacts:
        return result.artifacts["explained_class"]

    return None


def _markdown_feature_list(top_features: pd.DataFrame) -> str:
    if top_features.empty:
        return "No top features were available."
    lines = []
    for _, row in top_features.head(10).iterrows():
        lines.append(
            f"- {row['feature']}: consensus_score={row['consensus_score']:.3f}; "
            f"{row['interpretation_hint']}"
        )
    return "\n".join(lines)
