"""Shared helper for saving notebook tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_dataframe_as_figure(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    include_index: bool = True,
    float_format: str = "{:.4f}",
    figure_width: float = 10.0,
    row_height: float = 0.55,
    min_height: float = 2.2,
    font_size: int = 10,
    header_color: str = "#d9ead3",
    dpi: int = 300,
) -> str:
    """Render a DataFrame as a static table image and save it to disk."""

    export_df = frame.reset_index() if include_index else frame.copy()
    export_df = export_df.copy()

    float_cols = export_df.select_dtypes(include="float").columns.tolist()
    for col in float_cols:
        export_df[col] = export_df[col].map(lambda value: float_format.format(value))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(figure_width, max(min_height, row_height * (len(export_df) + 1)))
    )
    ax.axis("off")

    table = ax.table(
        cellText=export_df.values,
        colLabels=export_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.35)

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor(header_color)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)
