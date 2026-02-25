# -*- coding: utf-8 -*-
"""
Feature engineering module.
Extracts metadata features from images and entity names for analysis.
"""

import os
import logging
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter

from src.constants import ENTITY_UNIT_MAP, DIMENSION_ENTITIES, WEIGHT_ENTITIES

logger = logging.getLogger(__name__)


def extract_image_metadata(image_path: str) -> dict:
    """Extract basic image metadata (size, format, aspect ratio)."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return {
                "img_width": width,
                "img_height": height,
                "img_aspect_ratio": round(width / height, 3) if height > 0 else 0,
                "img_format": img.format or "unknown",
                "img_mode": img.mode,
            }
    except Exception:
        return {
            "img_width": 0, "img_height": 0,
            "img_aspect_ratio": 0, "img_format": "error", "img_mode": "error",
        }


def add_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features based on entity_name."""
    df = df.copy()
    df["entity_category"] = df["entity_name"].apply(get_entity_category)
    df["num_allowed_units"] = df["entity_name"].apply(
        lambda x: len(ENTITY_UNIT_MAP.get(x, set()))
    )
    return df


def get_entity_category(entity_name: str) -> str:
    """Map entity_name to a broader category."""
    if entity_name in DIMENSION_ENTITIES:
        return "dimension"
    elif entity_name in WEIGHT_ENTITIES:
        return "weight"
    elif entity_name in {"voltage", "wattage"}:
        return "electrical"
    elif entity_name in {"item_volume"}:
        return "volume"
    return "other"


def build_features(df: pd.DataFrame, image_dir: str = None) -> pd.DataFrame:
    """
    Build feature-enriched DataFrame with image metadata and entity features.

    Args:
        df: Input DataFrame with image_link and entity_name columns
        image_dir: Directory where images are stored

    Returns:
        DataFrame with added feature columns
    """
    df = add_entity_features(df)

    if image_dir and "image_link" in df.columns:
        logger.info("Extracting image metadata features...")
        metadata_records = []
        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, row["image_link"].split("/")[-1])
            metadata_records.append(extract_image_metadata(img_path))
        metadata_df = pd.DataFrame(metadata_records)
        df = pd.concat([df.reset_index(drop=True), metadata_df], axis=1)

    return df


def analyze_entity_distribution(df: pd.DataFrame) -> dict:
    """Analyze the distribution of entity types in the dataset."""
    dist = Counter(df["entity_name"].tolist())
    total = len(df)
    analysis = {
        entity: {"count": count, "percentage": round(count / total * 100, 2)}
        for entity, count in dist.most_common()
    }
    return analysis
