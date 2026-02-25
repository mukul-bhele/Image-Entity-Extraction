# -*- coding: utf-8 -*-
"""
Data preparation pipeline.
Downloads images and prepares train/test splits for the entity extraction task.
"""

import os
import logging
import pandas as pd
from pathlib import Path

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_IMAGE_DIR, TEST_IMAGE_DIR,
    TRAIN_CSV, TEST_CSV, FEW_SHOT_EXEMPLAR_DIR, NUM_FEW_SHOT_EXAMPLES,
)
from src.utils import download_images, get_local_image_path
from src.constants import ENTITY_UNIT_MAP

logger = logging.getLogger(__name__)


def prepare_directories():
    """Create all required data directories."""
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAIN_IMAGE_DIR,
              TEST_IMAGE_DIR, FEW_SHOT_EXEMPLAR_DIR]:
        os.makedirs(d, exist_ok=True)
    logger.info("Data directories created.")


def download_train_images(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """Download training images and add local_path column."""
    urls = df["image_link"].tolist()
    url_to_path = download_images(urls, str(TRAIN_IMAGE_DIR), max_workers=max_workers)
    df = df.copy()
    df["local_image_path"] = df["image_link"].apply(
        lambda u: url_to_path.get(u, get_local_image_path(u, str(TRAIN_IMAGE_DIR)))
    )
    # Filter out rows where download failed
    df["image_exists"] = df["local_image_path"].apply(
        lambda p: p is not None and os.path.exists(p)
    )
    failed = (~df["image_exists"]).sum()
    if failed > 0:
        logger.warning(f"{failed} training images failed to download.")
    df = df[df["image_exists"]].drop(columns=["image_exists"])
    return df


def download_test_images(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """Download test images and add local_path column."""
    urls = df["image_link"].tolist()
    url_to_path = download_images(urls, str(TEST_IMAGE_DIR), max_workers=max_workers)
    df = df.copy()
    df["local_image_path"] = df["image_link"].apply(
        lambda u: url_to_path.get(u, get_local_image_path(u, str(TEST_IMAGE_DIR)))
    )
    df["image_exists"] = df["local_image_path"].apply(
        lambda p: p is not None and os.path.exists(p)
    )
    failed = (~df["image_exists"]).sum()
    if failed > 0:
        logger.warning(f"{failed} test images failed to download.")
    df = df[df["image_exists"]].drop(columns=["image_exists"])
    return df


def build_few_shot_exemplars(train_df: pd.DataFrame):
    """
    Build few-shot exemplar pools segmented by entity_name and group_id.
    Saves curated exemplar CSVs for each entity type.
    """
    os.makedirs(str(FEW_SHOT_EXEMPLAR_DIR), exist_ok=True)

    for entity_name in ENTITY_UNIT_MAP.keys():
        entity_df = train_df[train_df["entity_name"] == entity_name].copy()
        if entity_df.empty:
            logger.warning(f"No training samples for entity: {entity_name}")
            continue

        # Filter to rows with non-empty entity_value
        entity_df = entity_df[entity_df["entity_value"].notna() & (entity_df["entity_value"] != "")]

        # Sample exemplars per group_id for diversity
        exemplars = []
        for group_id, group_df in entity_df.groupby("group_id"):
            n_samples = min(NUM_FEW_SHOT_EXAMPLES, len(group_df))
            sampled = group_df.sample(n=n_samples, random_state=42)
            exemplars.append(sampled)

        if exemplars:
            exemplar_df = pd.concat(exemplars, ignore_index=True)
            save_path = FEW_SHOT_EXEMPLAR_DIR / f"{entity_name}_exemplars.csv"
            exemplar_df.to_csv(save_path, index=False)
            logger.info(f"Saved {len(exemplar_df)} exemplars for {entity_name}")


def make_dataset(download_images_flag: bool = True, max_workers: int = 8):
    """
    Main data preparation entrypoint.
    1. Ensures directories exist
    2. Loads CSV files
    3. Downloads images (if flag set)
    4. Builds few-shot exemplar pools
    5. Saves processed CSVs
    """
    prepare_directories()

    # Load raw CSVs
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Training CSV not found at {TRAIN_CSV}. "
            f"Please place train.csv in {RAW_DATA_DIR}"
        )

    train_df = pd.read_csv(TRAIN_CSV)
    logger.info(f"Loaded training data: {len(train_df)} samples")

    test_df = None
    if TEST_CSV.exists():
        test_df = pd.read_csv(TEST_CSV)
        logger.info(f"Loaded test data: {len(test_df)} samples")

    # Download images
    if download_images_flag:
        logger.info("Downloading training images...")
        train_df = download_train_images(train_df, max_workers=max_workers)

        if test_df is not None:
            logger.info("Downloading test images...")
            test_df = download_test_images(test_df, max_workers=max_workers)

    # Build few-shot exemplars from training data
    logger.info("Building few-shot exemplar pools...")
    build_few_shot_exemplars(train_df)

    # Save processed CSVs
    train_processed_path = PROCESSED_DATA_DIR / "train_processed.csv"
    train_df.to_csv(train_processed_path, index=False)
    logger.info(f"Saved processed training data to {train_processed_path}")

    if test_df is not None:
        test_processed_path = PROCESSED_DATA_DIR / "test_processed.csv"
        test_df.to_csv(test_processed_path, index=False)
        logger.info(f"Saved processed test data to {test_processed_path}")

    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    make_dataset()
