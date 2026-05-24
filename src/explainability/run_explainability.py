"""High-level orchestration for explainability artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from explainability.base_explainer import BaseExplainer, ExplanationResult
from explainability.reporting import (
    consolidate_feature_rankings,
    save_explainability_report,
)
from explainability.visualizations import plot_pdp_for_top_features


def generate_explainability_artifacts(
    explainer: BaseExplainer,
    x: pd.DataFrame,
    y: Any | None = None,
    *,
    instance_indices: list[int] | None = None,
    output_dir: str | Path = "reports",
) -> dict[str, Any]:
    """Generate global, local, tabular, visual, and report artifacts."""

    explainer.config["output_dir"] = str(output_dir)
    instance_indices = instance_indices or [0]

    global_result = explainer.explain_global(x, y)
    top_features = consolidate_feature_rankings(
        global_result,
        top_n=int(explainer.config.get("top_n_features", 10)),
        output_dir=output_dir,
    )

    local_results: list[ExplanationResult] = []
    for instance_index in instance_indices:
        local_results.append(explainer.explain_local(x, instance_index))

    figures = _collect_paths("figures", global_result, local_results)
    tables = _collect_paths("tables", global_result, local_results)

    pdp_path = _maybe_generate_pdp(
        explainer,
        x,
        top_features,
        output_dir=output_dir,
    )
    if pdp_path is not None:
        figures["pdp_top_features"] = pdp_path

    report_path = save_explainability_report(
        model_name=global_result.model_type,
        global_result=global_result,
        local_results=local_results,
        top_features=top_features,
        figures=figures,
        tables=tables,
        instance_indices=instance_indices,
        output_dir=output_dir,
    )

    return {
        "global_result": global_result,
        "local_results": local_results,
        "top_features": top_features,
        "figures": figures,
        "tables": tables,
        "report": report_path,
        "output_dir": str(output_dir),
    }


def _collect_paths(
    attribute: str,
    global_result: ExplanationResult,
    local_results: list[ExplanationResult],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for prefix, result in [("global", global_result)] + [
        (f"local_{idx}", result) for idx, result in enumerate(local_results)
    ]:
        values = getattr(result, attribute) or {}
        for name, path in values.items():
            paths[f"{prefix}_{name}"] = path
    return paths


def _maybe_generate_pdp(
    explainer: BaseExplainer,
    x: pd.DataFrame,
    top_features: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> str | None:
    if top_features.empty:
        return None
    top_n = int(explainer.config.get("pdp_top_n", 3))
    grid_resolution = int(explainer.config.get("pdp_grid_resolution", 50))
    x_df = explainer._as_dataframe(x)
    original_features = set(explainer._input_feature_names(x_df))
    features = [
        feature
        for feature in top_features["feature"].astype(str).tolist()
        if feature in original_features
    ][:top_n]
    if not features:
        return None
    try:
        return plot_pdp_for_top_features(
            explainer.pipeline,
            x_df,
            features,
            output_dir=output_dir,
            grid_resolution=grid_resolution,
        )
    except Exception:
        return None
