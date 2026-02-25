# -*- coding: utf-8 -*-
"""
Training pipeline for Qwen2-VL-7B using QLoRA via LLaMA-Factory.
Prepares training data in the required format and launches fine-tuning.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path

from src.config import (
    QWEN2_MODEL_NAME, FINETUNE_CONFIG, PROCESSED_DATA_DIR,
    TRAIN_IMAGE_DIR, MODELS_DIR,
)
from src.constants import ENTITY_UNIT_MAP

logger = logging.getLogger(__name__)


def prepare_training_data(
    train_df: pd.DataFrame,
    output_path: str = None,
    max_samples: int = None,
) -> str:
    """
    Convert training DataFrame to LLaMA-Factory compatible JSON format.

    Each sample becomes a conversation with image + prompt -> entity_value.

    Args:
        train_df: DataFrame with columns [image_link, entity_name, entity_value]
        output_path: Path to save the JSON file
        max_samples: Maximum number of samples to use

    Returns:
        Path to the saved JSON file
    """
    if output_path is None:
        output_path = str(PROCESSED_DATA_DIR / "train_llama_factory.json")

    if max_samples is None:
        max_samples = FINETUNE_CONFIG.get("max_samples", 150_000)

    # Filter valid samples
    df = train_df.dropna(subset=["entity_value"]).copy()
    df = df[df["entity_value"].str.strip() != ""]

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        logger.info(f"Downsampled training data to {max_samples} samples")

    training_data = []
    for _, row in df.iterrows():
        entity_name = row["entity_name"]
        entity_value = str(row["entity_value"]).strip()
        readable_entity = " ".join(entity_name.split("_"))
        allowed_units = sorted(ENTITY_UNIT_MAP.get(entity_name, set()))

        image_filename = str(row["image_link"]).split("/")[-1]
        image_path = os.path.join(str(TRAIN_IMAGE_DIR), image_filename)

        # Parse entity_value into value + unit
        parts = entity_value.split(" ", 1)
        if len(parts) == 2:
            value, unit = parts
            answer = f"Value: {value}\nUnit: {unit}"
        else:
            answer = f"Value: {entity_value}\nUnit: unknown"

        sample = {
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        f"<image>\nExtract the {readable_entity} and its unit of "
                        f"measurement from the image.\n"
                        f"Allowed units: {', '.join(allowed_units)}\n"
                        f"Return ONLY in format:\nValue: <number>\nUnit: <unit>"
                    ),
                },
                {
                    "from": "gpt",
                    "value": answer,
                },
            ],
            "images": [image_path],
        }
        training_data.append(sample)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(training_data)} training samples to {output_path}")
    return output_path


def generate_llamafactory_config(training_data_path: str) -> str:
    """
    Generate a LLaMA-Factory YAML config for QLoRA fine-tuning.

    Returns:
        Path to the generated config file
    """
    config = {
        "model_name_or_path": QWEN2_MODEL_NAME,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": FINETUNE_CONFIG["lora_rank"],
        "lora_alpha": FINETUNE_CONFIG["lora_alpha"],
        "lora_dropout": FINETUNE_CONFIG["lora_dropout"],
        "quantization_bit": FINETUNE_CONFIG["quantization_bit"],
        "dataset": "custom_entity_extraction",
        "dataset_dir": str(PROCESSED_DATA_DIR),
        "template": "qwen2_vl",
        "output_dir": FINETUNE_CONFIG["output_dir"],
        "per_device_train_batch_size": FINETUNE_CONFIG["per_device_train_batch_size"],
        "num_train_epochs": FINETUNE_CONFIG["num_train_epochs"],
        "learning_rate": FINETUNE_CONFIG["learning_rate"],
        "logging_steps": FINETUNE_CONFIG["logging_steps"],
        "save_steps": FINETUNE_CONFIG["save_steps"],
        "bf16": True,
        "gradient_accumulation_steps": 1,
        "preprocessing_num_workers": 4,
    }

    # Write LLaMA-Factory dataset_info.json
    dataset_info = {
        "custom_entity_extraction": {
            "file_name": os.path.basename(training_data_path),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
            },
        }
    }
    dataset_info_path = str(PROCESSED_DATA_DIR / "dataset_info.json")
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Write training config YAML
    config_path = str(MODELS_DIR / "finetune_config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Generated LLaMA-Factory config at {config_path}")
    return config_path


def train(train_df: pd.DataFrame = None):
    """
    Main training entrypoint.
    1. Loads or receives training data
    2. Prepares LLaMA-Factory format
    3. Generates config
    4. Launches training via CLI
    """
    if train_df is None:
        train_processed_path = PROCESSED_DATA_DIR / "train_processed.csv"
        if not train_processed_path.exists():
            raise FileNotFoundError(
                f"Processed training data not found at {train_processed_path}. "
                f"Run data preparation first."
            )
        train_df = pd.read_csv(train_processed_path)

    # Step 1: Prepare training data
    logger.info("Preparing training data for LLaMA-Factory...")
    training_data_path = prepare_training_data(train_df)

    # Step 2: Generate config
    logger.info("Generating LLaMA-Factory config...")
    config_path = generate_llamafactory_config(training_data_path)

    # Step 3: Launch training
    logger.info("Launching QLoRA fine-tuning...")
    logger.info(
        f"Run the following command to start training:\n"
        f"  llamafactory-cli train {config_path}"
    )

    # Attempt to launch via subprocess
    import subprocess
    try:
        result = subprocess.run(
            ["llamafactory-cli", "train", config_path],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Training completed successfully.")
        logger.info(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    except FileNotFoundError:
        logger.warning(
            "llamafactory-cli not found. Install LLaMA-Factory:\n"
            "  pip install llamafactory\n"
            f"Then run: llamafactory-cli train {config_path}"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e.stderr}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    train()
