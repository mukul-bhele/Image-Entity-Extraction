# -*- coding: utf-8 -*-
"""
Qwen2-VL-7B Zero-Shot Prompting (ZSP) module.
Uses HuggingFace Transformers for inference with flash attention.
"""

import os
import logging
import torch
import pandas as pd
from tqdm import tqdm

from src.config import (
    QWEN2_MODEL_NAME, BATCH_SIZE, MAX_NEW_TOKENS,
    QWEN2_MIN_PIXELS, QWEN2_MAX_PIXELS,
    QWEN2_PROMPT_TEMPLATE, QWEN2_ZSP_OUTPUT_CSV, TEST_IMAGE_DIR,
)
from src.constants import ENTITY_UNIT_MAP
from src.utils import parse_model_response, post_process_prediction

logger = logging.getLogger(__name__)


def build_prompt(entity_name: str) -> str:
    """Build a zero-shot prompt for Qwen2-VL."""
    readable_entity = " ".join(entity_name.split("_"))
    return QWEN2_PROMPT_TEMPLATE.format(entity_name=readable_entity)


def build_messages(image_path: str, entity_name: str) -> list:
    """Build the chat message format for Qwen2-VL."""
    prompt_text = build_prompt(entity_name)
    allowed_units = sorted(ENTITY_UNIT_MAP.get(entity_name, set()))
    unit_hint = f"\nAllowed units: {', '.join(allowed_units)}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt_text + unit_hint},
            ],
        }
    ]
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
    logger.info(f"Loaded Qwen2-VL model: {QWEN2_MODEL_NAME}")
    return model, processor


def generate_single(model, processor, messages: list) -> str:
    """Run inference for a single image+prompt through Qwen2-VL."""
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

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] if output_text else ""


def predict_qwen2_zsp(
    test_df: pd.DataFrame,
    image_dir: str = None,
) -> pd.DataFrame:
    """
    Run zero-shot inference with Qwen2-VL-7B on test data.

    Args:
        test_df: DataFrame with columns [index, image_link, entity_name]
        image_dir: Directory containing downloaded test images

    Returns:
        DataFrame with columns [index, prediction]
    """
    if image_dir is None:
        image_dir = str(TEST_IMAGE_DIR)

    model, processor = load_qwen2_model()

    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Qwen2 ZSP"):
        torch.cuda.empty_cache()

        image_path = os.path.join(image_dir, row["image_link"].split("/")[-1])
        entity_name = row["entity_name"]

        messages = build_messages(image_path, entity_name)
        raw_response = generate_single(model, processor, messages)

        parsed = parse_model_response(raw_response, entity_name)
        processed = post_process_prediction(parsed, entity_name)
        predictions.append({"index": row["index"], "prediction": processed})

    result_df = pd.DataFrame(predictions)

    os.makedirs(os.path.dirname(str(QWEN2_ZSP_OUTPUT_CSV)), exist_ok=True)
    result_df.to_csv(QWEN2_ZSP_OUTPUT_CSV, index=False)
    logger.info(f"Saved Qwen2 ZSP predictions to {QWEN2_ZSP_OUTPUT_CSV}")

    return result_df
