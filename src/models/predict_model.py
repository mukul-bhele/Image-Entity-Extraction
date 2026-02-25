# -*- coding: utf-8 -*-
"""
Prediction orchestrator.
Runs inference with one or more model strategies and produces the final output.
"""

import os
import logging
import pandas as pd
from pathlib import Path

from src.config import (
    PROCESSED_DATA_DIR, TEST_IMAGE_DIR, FINAL_OUTPUT_CSV, OUTPUT_DIR,
    MINICPM_OUTPUT_CSV, QWEN2_ZSP_OUTPUT_CSV,
    QWEN2_FSL_OUTPUT_CSV, QWEN2_SFT_OUTPUT_CSV,
)
from src.utils import sanity_check

logger = logging.getLogger(__name__)


def load_test_data() -> pd.DataFrame:
    """Load the processed test DataFrame."""
    test_path = PROCESSED_DATA_DIR / "test_processed.csv"
    if not test_path.exists():
        from src.config import TEST_CSV
        if TEST_CSV.exists():
            return pd.read_csv(TEST_CSV)
        raise FileNotFoundError(
            f"Test data not found. Run data preparation first."
        )
    return pd.read_csv(test_path)


def predict_single_model(model_name: str, test_df: pd.DataFrame) -> pd.DataFrame:
    """Run prediction with a single specified model."""
    if model_name == "minicpm_zsp":
        from src.models.minicpm_zsp import predict_minicpm
        return predict_minicpm(test_df)

    elif model_name == "qwen2_zsp":
        from src.models.qwen2_zsp import predict_qwen2_zsp
        return predict_qwen2_zsp(test_df)

    elif model_name == "qwen2_fsl":
        from src.models.qwen2_fsl import predict_qwen2_fsl
        return predict_qwen2_fsl(test_df)

    elif model_name == "qwen2_sft":
        from src.models.qwen2_sft import predict_qwen2_sft
        return predict_qwen2_sft(test_df)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def predict_ensemble(
    test_df: pd.DataFrame,
    models: list = None,
) -> pd.DataFrame:
    """
    Run predictions with multiple models and ensemble the results.

    Args:
        test_df: Test DataFrame
        models: List of model names to use. Defaults to all three strategies.

    Returns:
        Ensembled predictions DataFrame
    """
    from src.models.ensemble import ensemble_predictions

    if models is None:
        models = ["minicpm_zsp", "qwen2_fsl", "qwen2_sft"]

    prediction_dfs = []
    for model_name in models:
        logger.info(f"Running predictions with {model_name}...")

        # Check for cached predictions
        cache_paths = {
            "minicpm_zsp": MINICPM_OUTPUT_CSV,
            "qwen2_zsp": QWEN2_ZSP_OUTPUT_CSV,
            "qwen2_fsl": QWEN2_FSL_OUTPUT_CSV,
            "qwen2_sft": QWEN2_SFT_OUTPUT_CSV,
        }
        cache_path = cache_paths.get(model_name)
        if cache_path and Path(cache_path).exists():
            logger.info(f"Loading cached predictions from {cache_path}")
            pred_df = pd.read_csv(cache_path)
        else:
            pred_df = predict_single_model(model_name, test_df)

        prediction_dfs.append(pred_df)

    # Ensemble
    logger.info(f"Ensembling {len(prediction_dfs)} model predictions...")
    result_df = ensemble_predictions(prediction_dfs)

    return result_df


def generate_final_output(
    predictions_df: pd.DataFrame,
    test_csv_path: str = None,
) -> pd.DataFrame:
    """
    Generate the final submission CSV and run sanity checks.

    Args:
        predictions_df: DataFrame with [index, prediction] columns
        test_csv_path: Path to original test CSV for validation

    Returns:
        Final validated DataFrame
    """
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # Ensure all predictions are strings and NaN becomes ""
    predictions_df = predictions_df.copy()
    predictions_df["prediction"] = predictions_df["prediction"].fillna("").astype(str)

    # Save final output
    predictions_df.to_csv(FINAL_OUTPUT_CSV, index=False)
    logger.info(f"Saved final predictions to {FINAL_OUTPUT_CSV}")

    # Run sanity check
    if test_csv_path:
        passed, message = sanity_check(str(FINAL_OUTPUT_CSV), test_csv_path)
        if passed:
            logger.info(f"Sanity check: {message}")
        else:
            logger.error(f"Sanity check FAILED: {message}")

    return predictions_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    test_df = load_test_data()
    result = predict_ensemble(test_df)
    generate_final_output(result)
