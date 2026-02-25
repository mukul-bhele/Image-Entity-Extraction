# -*- coding: utf-8 -*-
"""
Visualization utilities for analysis and reporting.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

from src.config import REPORTS_DIR
from src.constants import ENTITY_UNIT_MAP

logger = logging.getLogger(__name__)


def plot_entity_distribution(df: pd.DataFrame, save_path: str = None):
    """Plot the distribution of entity types in the dataset."""
    fig, ax = plt.subplots(figsize=(12, 6))
    entity_counts = df["entity_name"].value_counts()
    entity_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Entity Type Distribution")
    ax.set_xlabel("Entity Name")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    if save_path is None:
        save_path = str(REPORTS_DIR / "figures" / "entity_distribution.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved entity distribution plot to {save_path}")


def plot_prediction_coverage(predictions_df: pd.DataFrame, save_path: str = None):
    """Plot the coverage of predictions (non-empty vs empty)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    non_empty = (predictions_df["prediction"].str.strip() != "").sum()
    empty = len(predictions_df) - non_empty

    ax.bar(["Predicted", "Empty"], [non_empty, empty], color=["green", "red"])
    ax.set_title("Prediction Coverage")
    ax.set_ylabel("Count")

    for i, v in enumerate([non_empty, empty]):
        pct = v / len(predictions_df) * 100
        ax.text(i, v + len(predictions_df) * 0.01, f"{v}\n({pct:.1f}%)", ha="center")

    plt.tight_layout()

    if save_path is None:
        save_path = str(REPORTS_DIR / "figures" / "prediction_coverage.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved prediction coverage plot to {save_path}")


def plot_unit_distribution(predictions_df: pd.DataFrame, save_path: str = None):
    """Plot the distribution of predicted units."""
    units = []
    for pred in predictions_df["prediction"]:
        if pd.notna(pred) and str(pred).strip():
            parts = str(pred).strip().split(" ", 1)
            if len(parts) == 2:
                units.append(parts[1])

    if not units:
        logger.warning("No non-empty predictions to plot unit distribution.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    unit_counts = Counter(units)
    unit_df = pd.DataFrame(
        unit_counts.most_common(20),
        columns=["unit", "count"],
    )
    sns.barplot(data=unit_df, x="unit", y="count", ax=ax, palette="viridis")
    ax.set_title("Top 20 Predicted Units")
    ax.set_xlabel("Unit")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    if save_path is None:
        save_path = str(REPORTS_DIR / "figures" / "unit_distribution.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved unit distribution plot to {save_path}")


def plot_f1_comparison(scores: dict, save_path: str = None):
    """Plot F1 score comparison across different model strategies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    models = list(scores.keys())
    f1_scores = list(scores.values())

    bars = ax.bar(models, f1_scores, color=sns.color_palette("Set2", len(models)))
    ax.set_title("F1 Score Comparison Across Strategies")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1)

    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{score:.3f}", ha="center", fontsize=10,
        )

    plt.tight_layout()

    if save_path is None:
        save_path = str(REPORTS_DIR / "figures" / "f1_comparison.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved F1 comparison plot to {save_path}")
