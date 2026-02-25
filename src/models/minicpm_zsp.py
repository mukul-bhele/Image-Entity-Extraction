# -*- coding: utf-8 -*-
"""
MiniCPM-V-2.6 Zero-Shot Prompting (ZSP) module.
Uses LMDeploy TurboMind for efficient batched inference.
"""

import os
import logging
import pickle
import torch
import pandas as pd
from tqdm import tqdm

from src.config import (
    MINICPM_MODEL_NAME, MINICPM_DOWNLOAD_DIR, BATCH_SIZE,
    MINICPM_PROMPT_TEMPLATE, MINICPM_OUTPUT_CSV, TEST_IMAGE_DIR,
)
from src.utils import parse_model_response, post_process_prediction

logger = logging.getLogger(__name__)


def build_prompt(entity_name: str) -> str:
    """Build a zero-shot prompt for MiniCPM given an entity name."""
    readable_entity = " ".join(entity_name.split("_"))
    return MINICPM_PROMPT_TEMPLATE.format(entity_name=readable_entity)


def load_minicpm_pipeline():
    """Load MiniCPM-V-2.6 model via LMDeploy pipeline."""
    from lmdeploy import pipeline, TurbomindEngineConfig

    os.makedirs(MINICPM_DOWNLOAD_DIR, exist_ok=True)
    pipe = pipeline(
        MINICPM_MODEL_NAME,
        backend_config=TurbomindEngineConfig(session_len=None),
    )
    logger.info(f"Loaded MiniCPM model: {MINICPM_MODEL_NAME}")
    return pipe


def predict_minicpm(
    test_df: pd.DataFrame,
    image_dir: str = None,
    batch_size: int = BATCH_SIZE,
    save_intermediate: bool = True,
) -> pd.DataFrame:
    """
    Run zero-shot inference with MiniCPM-V-2.6 on test data.

    Args:
        test_df: DataFrame with columns [index, image_link, entity_name]
        image_dir: Directory containing downloaded test images
        batch_size: Number of images to process per batch
        save_intermediate: Whether to save raw responses as pickle

    Returns:
        DataFrame with columns [index, prediction]
    """
    if image_dir is None:
        image_dir = str(TEST_IMAGE_DIR)

    pipe = load_minicpm_pipeline()

    # Prepare image paths and prompts
    images = test_df["image_link"].apply(
        lambda x: os.path.join(image_dir, x.split("/")[-1])
    ).tolist()
    entity_names = test_df["entity_name"].tolist()
    prompts = [build_prompt(en) for en in entity_names]
    indices = test_df["index"].tolist()

    # Batch inference
    raw_responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="MiniCPM ZSP"):
        torch.cuda.empty_cache()
        batch_end = min(i + batch_size, len(prompts))
        batch = [(prompts[j], images[j]) for j in range(i, batch_end)]

        responses = pipe(batch)
        for j in range(len(responses)):
            raw_responses.append(responses[j].text)

    # Save raw responses
    if save_intermediate:
        os.makedirs(os.path.dirname(str(MINICPM_OUTPUT_CSV)), exist_ok=True)
        raw_path = str(MINICPM_OUTPUT_CSV).replace(".csv", "_raw.pkl")
        with open(raw_path, "wb") as f:
            pickle.dump(raw_responses, f)
        logger.info(f"Saved raw MiniCPM responses to {raw_path}")

    # Parse and post-process responses
    predictions = []
    for idx, response, entity_name in zip(indices, raw_responses, entity_names):
        parsed = parse_model_response(response, entity_name)
        processed = post_process_prediction(parsed, entity_name)
        predictions.append({"index": idx, "prediction": processed})

    result_df = pd.DataFrame(predictions)

    # Save predictions
    os.makedirs(os.path.dirname(str(MINICPM_OUTPUT_CSV)), exist_ok=True)
    result_df.to_csv(MINICPM_OUTPUT_CSV, index=False)
    logger.info(f"Saved MiniCPM predictions to {MINICPM_OUTPUT_CSV}")

    return result_df
