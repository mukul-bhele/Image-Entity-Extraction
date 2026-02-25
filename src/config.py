# -*- coding: utf-8 -*-
"""
Central configuration for the Image Entity Extraction pipeline.
All paths, model settings, and hyperparameters are defined here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGE_DIR = DATA_DIR / "images"
TRAIN_IMAGE_DIR = IMAGE_DIR / "train"
TEST_IMAGE_DIR = IMAGE_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "output"

# CSV paths
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
SAMPLE_OUTPUT_CSV = RAW_DATA_DIR / "sample_test_out.csv"

# ──────────────────────────────────────────────────────────────────────
# Model configurations
# ──────────────────────────────────────────────────────────────────────

# MiniCPM-V-2.6 (Zero-Shot Prompting)
MINICPM_MODEL_NAME = "openbmb/MiniCPM-V-2_6"
MINICPM_DOWNLOAD_DIR = str(MODELS_DIR / "minicpm")

# Qwen2-VL-7B (Few-Shot + Fine-Tuned)
QWEN2_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
QWEN2_FINETUNED_PATH = str(MODELS_DIR / "qwen2_finetuned")

# ──────────────────────────────────────────────────────────────────────
# Inference settings
# ──────────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
MAX_NEW_TOKENS = 128
NUM_WORKERS = 4

# GPU settings
CUDA_DEVICE = 0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(CUDA_DEVICE))

# Qwen2 processor pixel constraints
QWEN2_MIN_PIXELS = 256 * 28 * 28
QWEN2_MAX_PIXELS = 1280 * 28 * 28

# ──────────────────────────────────────────────────────────────────────
# Fine-tuning settings (QLoRA via LLaMA-Factory)
# ──────────────────────────────────────────────────────────────────────
FINETUNE_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 16,
    "learning_rate": 1e-4,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "quantization_bit": 8,       # 8-bit QLoRA
    "max_samples": 150_000,
    "output_dir": str(MODELS_DIR / "qwen2_finetuned"),
    "logging_steps": 50,
    "save_steps": 500,
}

# ──────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────
MINICPM_PROMPT_TEMPLATE = (
    "Extract the best (only one) {entity_name} and its unit from the image "
    "in the format -\nValue: <only numbers>\nUnit: <only alphabets>"
)

QWEN2_PROMPT_TEMPLATE = (
    "Extract the {entity_name} and its unit of measurement from the image. "
    "Return ONLY the value and unit in this exact format:\n"
    "Value: <number>\nUnit: <unit>\n"
    "If no {entity_name} is found, return: None"
)

# ──────────────────────────────────────────────────────────────────────
# Few-shot exemplar settings
# ──────────────────────────────────────────────────────────────────────
NUM_FEW_SHOT_EXAMPLES = 3
FEW_SHOT_EXEMPLAR_DIR = PROCESSED_DATA_DIR / "exemplars"

# ──────────────────────────────────────────────────────────────────────
# Ensemble settings
# ──────────────────────────────────────────────────────────────────────
ENSEMBLE_STRATEGY = "majority_vote"  # Options: majority_vote, confidence_weighted

# ──────────────────────────────────────────────────────────────────────
# Output settings
# ──────────────────────────────────────────────────────────────────────
MINICPM_OUTPUT_CSV = OUTPUT_DIR / "minicpm_predictions.csv"
QWEN2_ZSP_OUTPUT_CSV = OUTPUT_DIR / "qwen2_zsp_predictions.csv"
QWEN2_FSL_OUTPUT_CSV = OUTPUT_DIR / "qwen2_fsl_predictions.csv"
QWEN2_SFT_OUTPUT_CSV = OUTPUT_DIR / "qwen2_sft_predictions.csv"
ENSEMBLE_OUTPUT_CSV = OUTPUT_DIR / "ensemble_predictions.csv"
FINAL_OUTPUT_CSV = OUTPUT_DIR / "test_out.csv"
