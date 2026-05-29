import matplotlib.pyplot as plt
import pandas as pd

from explainability.visualizations import (
    plot_lime_explanation,
    plot_native_feature_importance,
    plot_permutation_importance,
)
from visualization_utils import save_dataframe_as_figure


class FakeLimeExplanation:
    def save_to_file(self, path):
        with open(path, "w", encoding="utf-8") as file:
            file.write("<html></html>")

    def as_list(self):
        return [("a <= 1", 0.4), ("b > 2", -0.2)]


def test_visualization_functions_create_files_and_close_figures(tmp_path):
    permutation = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "importance_mean": [0.2, 0.1],
            "importance_std": [0.01, 0.02],
        }
    )
    native = pd.DataFrame({"feature": ["a", "b"], "importance": [0.7, 0.3]})

    png_1 = plot_permutation_importance(permutation, output_dir=tmp_path)
    png_2 = plot_native_feature_importance(native, output_dir=tmp_path)
    lime_paths = plot_lime_explanation(FakeLimeExplanation(), output_dir=tmp_path)

    assert (
        tmp_path / "figures" / "explainability" / "permutation_importance.png"
    ).exists()
    assert (
        tmp_path / "figures" / "explainability" / "native_feature_importance.png"
    ).exists()
    assert (tmp_path / "figures" / "explainability" / "lime_instance_0.html").exists()
    assert png_1.endswith(".png")
    assert png_2.endswith(".png")
    assert lime_paths["csv_path"].endswith(".csv")
    assert plt.get_fignums() == []


def test_save_dataframe_as_figure_creates_file_and_closes_figures(tmp_path):
    ranking = pd.DataFrame(
        {
            "Model": ["RF", "SVM"],
            "f1_mean": [0.9987, 0.9923],
            "f1_std": [0.0012, 0.0045],
        },
        index=pd.Index([1, 2], name="Rank"),
    )

    output = save_dataframe_as_figure(
        ranking,
        tmp_path / "reports" / "paper" / "figures" / "ranking.png",
    )

    assert (tmp_path / "reports" / "paper" / "figures" / "ranking.png").exists()
    assert output.endswith("ranking.png")
    assert plt.get_fignums() == []
