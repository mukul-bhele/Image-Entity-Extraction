# -*- coding: utf-8 -*-
"""
Qwen2-VL-7B Few-Shot Learning (FSL) module.
Dynamically selects exemplars per entity type for in-context learning.
"""

import os
import logging
import torch
import pandas as pd
from tqdm import tqdm

from src.config import (
    QWEN2_MODEL_NAME, MAX_NEW_TOKENS, NUM_FEW_SHOT_EXAMPLES,
    QWEN2_MIN_PIXELS, QWEN2_MAX_PIXELS,
    FEW_SHOT_EXEMPLAR_DIR, QWEN2_FSL_OUTPUT_CSV, TEST_IMAGE_DIR,
)
from src.constants import ENTITY_UNIT_MAP
from src.utils import parse_model_response, post_process_prediction

logger = logging.getLogger(__name__)


def load_exemplars(entity_name: str) -> pd.DataFrame:
    """Load few-shot exemplars for a given entity type."""
    exemplar_path = FEW_SHOT_EXEMPLAR_DIR / f"{entity_name}_exemplars.csv"
    if not exemplar_path.exists():
        logger.warning(f"No exemplar file for {entity_name}, using zero-shot fallback")
        return pd.DataFrame()
    return pd.read_csv(exemplar_path)


def select_exemplars(
    exemplar_df: pd.DataFrame,
    group_id: int = None,
    n: int = NUM_FEW_SHOT_EXAMPLES,
) -> pd.DataFrame:
    """
    Select n exemplars, prioritizing same group_id for relevance.
    Falls back to random sampling if not enough same-group examples.
    """
    if exemplar_df.empty:
        return exemplar_df

    if group_id is not None and "group_id" in exemplar_df.columns:
        same_group = exemplar_df[exemplar_df["group_id"] == group_id]
        if len(same_group) >= n:
            return same_group.sample(n=n, random_state=42)
        # Fill remaining from other groups
        other = exemplar_df[exemplar_df["group_id"] != group_id]
        remaining = n - len(same_group)
        if len(other) >= remaining:
            extra = other.sample(n=remaining, random_state=42)
        else:
            extra = other
        return pd.concat([same_group, extra], ignore_index=True).head(n)

    return exemplar_df.sample(n=min(n, len(exemplar_df)), random_state=42)


def build_few_shot_messages(
    image_path: str,
    entity_name: str,
    exemplars: pd.DataFrame,
    image_dir: str,
) -> list:
    """Build multi-turn chat messages with few-shot exemplars."""
    readable_entity = " ".join(entity_name.split("_"))
    allowed_units = sorted(ENTITY_UNIT_MAP.get(entity_name, set()))

    messages = []

    # Add exemplar turns
    for _, ex in exemplars.iterrows():
        ex_image_path = os.path.join(image_dir, str(ex["image_link"]).split("/")[-1])
        if not os.path.exists(ex_image_path):
            continue

        # Exemplar question
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{ex_image_path}"},
                {"type": "text", "text": (
                    f"Extract the {readable_entity} and its unit of measurement "
                    f"from the image.\nAllowed units: {', '.join(allowed_units)}\n"
                    f"Return ONLY in format:\nValue: <number>\nUnit: <unit>"
                )},
            ],
        })
        # Exemplar answer
        entity_value = str(ex.get("entity_value", ""))
        if entity_value and entity_value != "nan":
            parts = entity_value.split(" ", 1)
            if len(parts) == 2:
                messages.append({
                    "role": "assistant",
                    "content": f"Value: {parts[0]}\nUnit: {parts[1]}",
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": f"Value: {entity_value}\nUnit: unknown",
                })

    # Actual query
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": (
                f"Extract the {readable_entity} and its unit of measurement "
                f"from the image.\nAllowed units: {', '.join(allowed_units)}\n"
                f"Return ONLY in format:\nValue: <number>\nUnit: <unit>\n"
                f"If no {readable_entity} is found, return: None"
            )},
        ],
    })

    return messages


def load_qwen2_model():
    """Load Qwen2-VL-7B model and processor."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN2_MODEL_NAME,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        QWEN2_MODEL_NAME,
        min_pixels=QWEN2_MIN_PIXELS,
        max_pixels=QWEN2_MAX_PIXELS,
    )
    logger.info(f"Loaded Qwen2-VL model for FSL: {QWEN2_MODEL_NAME}")
    return model, processor


def generate_single(model, processor, messages: list) -> str:
    """Run inference for a single few-shot prompt through Qwen2-VL."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] if output_text else ""


def predict_qwen2_fsl(
    test_df: pd.DataFrame,
    image_dir: str = None,
    train_image_dir: str = None,
) -> pd.DataFrame:
    """
    Run few-shot inference with Qwen2-VL-7B on test data.

    Args:
        test_df: DataFrame with columns [index, image_link, entity_name, group_id]
        image_dir: Directory containing downloaded test images
        train_image_dir: Directory containing training images (for exemplars)

    Returns:
        DataFrame with columns [index, prediction]
    """
    from src.config import TRAIN_IMAGE_DIR

    if image_dir is None:
        image_dir = str(TEST_IMAGE_DIR)
    if train_image_dir is None:
        train_image_dir = str(TRAIN_IMAGE_DIR)

    model, processor = load_qwen2_model()

    # Pre-load all exemplar pools
    exemplar_pools = {}
    for entity_name in ENTITY_UNIT_MAP.keys():
        exemplar_pools[entity_name] = load_exemplars(entity_name)

    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Qwen2 FSL"):
        torch.cuda.empty_cache()

        image_path = os.path.join(image_dir, row["image_link"].split("/")[-1])
        entity_name = row["entity_name"]
        group_id = row.get("group_id", None)

        # Select exemplars
        exemplar_df = exemplar_pools.get(entity_name, pd.DataFrame())
        selected = select_exemplars(exemplar_df, group_id=group_id)

        # Build messages with few-shot examples
        messages = build_few_shot_messages(
            image_path, entity_name, selected, train_image_dir
        )

        raw_response = generate_single(model, processor, messages)
        parsed = parse_model_response(raw_response, entity_name)
        processed = post_process_prediction(parsed, entity_name)
        predictions.append({"index": row["index"], "prediction": processed})

    result_df = pd.DataFrame(predictions)

    os.makedirs(os.path.dirname(str(QWEN2_FSL_OUTPUT_CSV)), exist_ok=True)
    result_df.to_csv(QWEN2_FSL_OUTPUT_CSV, index=False)
    logger.info(f"Saved Qwen2 FSL predictions to {QWEN2_FSL_OUTPUT_CSV}")

    return result_df
