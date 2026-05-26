"""Plotting helpers for explainability figures and local explanation exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

DEFAULT_FIGURE_DIR = Path("reports") / "figures" / "explainability"


def _figure_dir(output_dir: str | Path | None = None) -> Path:
    base = Path(output_dir) if output_dir is not None else Path("reports")
    path = base / "figures" / "explainability"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_current(path: Path) -> str:
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def _top_frame(
    frame: pd.DataFrame,
    value_col: str,
    top_n: int,
    *,
    absolute: bool = False,
) -> pd.DataFrame:
    data = frame.copy()
    sort_col = f"abs_{value_col}" if absolute else value_col
    if absolute and sort_col not in data.columns:
        data[sort_col] = data[value_col].abs()
    return data.sort_values(sort_col, ascending=False).head(top_n).iloc[::-1]


def plot_permutation_importance(
    permutation_importance: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    top_n: int = 15,
    filename: str = "permutation_importance.png",
) -> str:
    """Save a horizontal bar chart of permutation feature importance."""

    data = _top_frame(permutation_importance, "importance_mean", top_n)
    plt.figure(figsize=(8, max(4, 0.35 * len(data))))
    plt.barh(data["feature"], data["importance_mean"], xerr=data.get("importance_std"))
    plt.xlabel("Permutation importance")
    plt.ylabel("Feature")
    plt.title("Permutation Feature Importance")
    return _save_current(_figure_dir(output_dir) / filename)


def plot_native_feature_importance(
    feature_importance: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    top_n: int = 15,
    filename: str = "native_feature_importance.png",
) -> str:
    """Save a horizontal bar chart of model-native feature importance."""

    data = _top_frame(feature_importance, "importance", top_n)
    plt.figure(figsize=(8, max(4, 0.35 * len(data))))
    plt.barh(data["feature"], data["importance"])
    plt.xlabel("Native importance")
    plt.ylabel("Feature")
    plt.title("Model-Specific Feature Importance")
    return _save_current(_figure_dir(output_dir) / filename)


def plot_linear_coefficients(
    coefficients: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    top_n: int = 15,
    signed: bool = False,
    filename: str | None = None,
) -> str:
    """Save a bar chart of signed or absolute linear coefficients."""

    value_col = "coefficient" if signed else "abs_coefficient"
    data = _top_frame(coefficients, value_col, top_n, absolute=signed)
    if filename is None:
        filename = (
            "coefficients_signed_bar.png" if signed else "coefficients_abs_bar.png"
        )
    colors = np.where(data["coefficient"] >= 0, "#2f6fbb", "#b64545")
    plt.figure(figsize=(8, max(4, 0.35 * len(data))))
    plt.barh(data["feature"], data[value_col], color=colors if signed else "#2f6fbb")
    plt.xlabel("Coefficient" if signed else "Absolute coefficient")
    plt.ylabel("Feature")
    plt.title("Linear Model Coefficients")
    return _save_current(_figure_dir(output_dir) / filename)


def plot_local_contributions(
    contributions: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    instance_index: int = 0,
    top_n: int = 15,
) -> str:
    """Save a bar chart of local linear feature contributions."""

    data = _top_frame(contributions, "abs_contribution", top_n)
    colors = np.where(data["contribution"] >= 0, "#2f6fbb", "#b64545")
    plt.figure(figsize=(8, max(4, 0.35 * len(data))))
    plt.barh(data["feature"], data["contribution"], color=colors)
    plt.xlabel("Local contribution")
    plt.ylabel("Feature")
    plt.title(f"Local Linear Contributions - Instance {instance_index}")
    return _save_current(
        _figure_dir(output_dir) / f"local_contributions_instance_{instance_index}.png"
    )


def plot_shap_global_bar(
    shap_values: Any,
    *,
    output_dir: str | Path | None = None,
    filename: str = "shap_global_bar.png",
) -> str:
    """Save a SHAP global importance bar plot."""

    import shap

    shap.plots.bar(shap_values, show=False)
    return _save_current(_figure_dir(output_dir) / filename)


def plot_shap_beeswarm(
    shap_values: Any,
    *,
    output_dir: str | Path | None = None,
    filename: str = "shap_beeswarm.png",
) -> str:
    """Save a SHAP beeswarm plot."""

    import shap

    shap.plots.beeswarm(shap_values, show=False)
    return _save_current(_figure_dir(output_dir) / filename)


def plot_shap_waterfall(
    shap_values: Any,
    *,
    output_dir: str | Path | None = None,
    instance_index: int = 0,
    filename: str | None = None,
) -> str:
    """Save a SHAP waterfall plot for one instance."""

    import shap

    if filename is None:
        filename = f"shap_waterfall_instance_{instance_index}.png"
    shap.plots.waterfall(shap_values, show=False)
    return _save_current(_figure_dir(output_dir) / filename)


def plot_lime_explanation(
    explanation: Any,
    *,
    output_dir: str | Path | None = None,
    instance_index: int = 0,
    label: int | None = None,
) -> dict[str, str]:
    """Save LIME explanation artifacts as HTML and CSV files."""

    figure_dir = _figure_dir(output_dir)
    html_path = figure_dir / f"lime_instance_{instance_index}.html"
    csv_path = figure_dir / f"lime_instance_{instance_index}.csv"
    explanation.save_to_file(str(html_path))
    as_list = (
        explanation.as_list(label=label) if label is not None else explanation.as_list()
    )
    pd.DataFrame(as_list, columns=["feature", "weight"]).to_csv(
        csv_path,
        index=False,
    )
    return {"html_path": str(html_path), "csv_path": str(csv_path)}


def plot_pdp_for_top_features(
    estimator: Any,
    x: pd.DataFrame,
    features: list[str] | list[int],
    *,
    output_dir: str | Path | None = None,
    grid_resolution: int = 50,
    filename: str = "pdp_top_features.png",
) -> str:
    """Save partial dependence plots for the selected input features."""

    if not features:
        raise ValueError("At least one feature is required to generate PDP plots.")

    PartialDependenceDisplay.from_estimator(
        estimator,
        x,
        features=features,
        grid_resolution=grid_resolution,
    )
    return _save_current(_figure_dir(output_dir) / filename)
