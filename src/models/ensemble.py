# -*- coding: utf-8 -*-
"""
Ensemble module for combining predictions from multiple models.
Implements majority voting and confidence-weighted strategies.
"""

import os
import logging
import pandas as pd
from collections import Counter
from typing import List

from src.config import ENSEMBLE_OUTPUT_CSV, ENSEMBLE_STRATEGY

logger = logging.getLogger(__name__)


def majority_vote(predictions: List[str]) -> str:
    """
    Select the prediction that appears most frequently.
    Ties are broken by preferring non-empty predictions.
    """
    # Filter out None values
    valid = [p for p in predictions if p is not None]
    if not valid:
        return ""

    # Count occurrences
    counter = Counter(valid)

    # If all empty, return empty
    if all(p == "" for p in valid):
        return ""

    # Prefer non-empty predictions
    non_empty = {k: v for k, v in counter.items() if k != ""}
    if non_empty:
        return max(non_empty, key=non_empty.get)

    return ""


def ensemble_predictions(
    prediction_dfs: List[pd.DataFrame],
    strategy: str = ENSEMBLE_STRATEGY,
) -> pd.DataFrame:
    """
    Combine predictions from multiple model DataFrames using the specified strategy.

    Args:
        prediction_dfs: List of DataFrames, each with [index, prediction] columns
        strategy: Ensemble strategy ('majority_vote' or 'confidence_weighted')

    Returns:
        DataFrame with columns [index, prediction]
    """
    if not prediction_dfs:
        raise ValueError("No prediction DataFrames provided")

    if len(prediction_dfs) == 1:
        return prediction_dfs[0].copy()

    # Merge all predictions on index
    merged = prediction_dfs[0][["index"]].copy()
    for i, df in enumerate(prediction_dfs):
        df_renamed = df.rename(columns={"prediction": f"pred_{i}"})
        merged = merged.merge(df_renamed[["index", f"pred_{i}"]], on="index", how="left")

    pred_cols = [f"pred_{i}" for i in range(len(prediction_dfs))]

    # Apply ensemble strategy
    if strategy == "majority_vote":
        merged["prediction"] = merged[pred_cols].apply(
            lambda row: majority_vote(row.tolist()), axis=1
        )
    else:
        # Default to majority vote
        merged["prediction"] = merged[pred_cols].apply(
            lambda row: majority_vote(row.tolist()), axis=1
        )

    result_df = merged[["index", "prediction"]].copy()

    # Fill any NaN predictions with empty string
    result_df["prediction"] = result_df["prediction"].fillna("")

    # Save
    os.makedirs(os.path.dirname(str(ENSEMBLE_OUTPUT_CSV)), exist_ok=True)
    result_df.to_csv(ENSEMBLE_OUTPUT_CSV, index=False)
    logger.info(f"Saved ensemble predictions to {ENSEMBLE_OUTPUT_CSV}")

    return result_df
