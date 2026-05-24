"""Gráficos descritivos a partir do CSV bruto de resultados.

Quatro funções que geram figuras matplotlib a partir do CSV bruto de resultados
(`results/raw.csv`) produzido por `src.pipeline.run_all`.

Cada função retorna `matplotlib.figure.Figure`. Se `output_path` for passado,
o gráfico também é salvo em PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.pipeline.regime import aggregate_by_regime


def _save_if_requested(fig: plt.Figure, output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")


def bar_metric_by_model(
    raw_results: pd.DataFrame,
    metric: str = "auc_ovo",
    output_path: Path | None = None,
) -> plt.Figure:
    """Gráfico de barras com média (+/- desvio) da métrica por modelo."""
    agg = (
        raw_results.groupby("model")[metric]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        agg.index,
        agg["mean"],
        yerr=agg["std"].fillna(0.0),
        capsize=4,
        color=sns.color_palette("muted", n_colors=len(agg)),
    )
    ax.set_ylabel(f"{metric} (média)")
    ax.set_xlabel("modelo")
    ax.set_title(f"Desempenho médio por modelo ({metric})")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def boxplot_metric_by_model(
    raw_results: pd.DataFrame,
    metric: str = "auc_ovo",
    output_path: Path | None = None,
) -> plt.Figure:
    """Boxplot da métrica por modelo (distribuição ao longo dos datasets)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    order = (
        raw_results.groupby("model")[metric]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    sns.boxplot(data=raw_results, x="model", y=metric, order=order, ax=ax)
    sns.stripplot(
        data=raw_results,
        x="model",
        y=metric,
        order=order,
        ax=ax,
        color="black",
        alpha=0.5,
        size=3,
    )
    ax.set_title(f"Distribuição de {metric} por modelo")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def bar_metric_by_regime(
    raw_results: pd.DataFrame,
    metadata: pd.DataFrame,
    regime_col: str = "regime_size",
    metric: str = "auc_ovo",
    output_path: Path | None = None,
) -> plt.Figure:
    """Barras agrupadas: média da métrica por regime e por modelo."""
    aggregated = aggregate_by_regime(
        raw_results, metadata, regime_col=regime_col, metric_col=metric
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=aggregated,
        x=regime_col,
        y="mean",
        hue="model",
        ax=ax,
    )
    ax.set_ylabel(f"{metric} (média)")
    ax.set_title(f"{metric} por {regime_col}")
    ax.legend(title="modelo", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig


def time_vs_quality(
    raw_results: pd.DataFrame,
    x: str = "total_time_s",
    y: str = "auc_ovo",
    output_path: Path | None = None,
) -> plt.Figure:
    """Scatter custo computacional (x) vs. desempenho (y), colorido por modelo."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=raw_results,
        x=x,
        y=y,
        hue="model",
        s=70,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel(f"{x} (escala log)")
    ax.set_ylabel(y)
    ax.set_title(f"Custo computacional vs. desempenho ({y} x {x})")
    ax.legend(title="modelo", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig
