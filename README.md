# Image Entity Extraction

Extraction of product entity values (weight, dimensions, voltage, volume, etc.) from product images using an ensemble of Vision-Language Models. Built for the **Amazon ML Challenge 2024** problem statement.

## Problem Statement

Given a product image and an entity name (e.g., `item_weight`, `voltage`), extract the corresponding value and unit (e.g., `"34 gram"`, `"220 volt"`) directly from the image. The model must return predictions in exact `"value unit"` format for evaluation via F1 score with exact string matching.

**Dataset**: ~230K training images, ~130K test images across 8 entity types.

## Architecture

The pipeline uses a **voting ensemble** of three complementary strategies:

```
Product Image + Entity Name
         |
         v
  +-----------------+     +------------------+     +------------------+
  | MiniCPM-V-2.6   |     | Qwen2-VL-7B      |     | Qwen2-VL-7B      |
  | Zero-Shot (ZSP)  |     | Few-Shot (FSL)   |     | Fine-Tuned (SFT)  |
  +-----------------+     +------------------+     +------------------+
         |                        |                        |
         v                        v                        v
  +--------------------------------------------------------------+
  |              Majority Vote Ensemble                          |
  +--------------------------------------------------------------+
         |
         v
  +--------------------------------------------------------------+
  |              Post-Processing Layer                            |
  |  - Fraction handling (1/2 -> 0.5)                            |
  |  - Symbol standardization (feet/inches)                      |
  |  - Range resolution (3-5 -> 5)                               |
  |  - Unit normalization (kg -> kilogram)                       |
  +--------------------------------------------------------------+
         |
         v
    Final Prediction: "value unit"
```

### Model Strategies

| Strategy | Model | Description |
|----------|-------|-------------|
| **Zero-Shot Prompting (ZSP)** | MiniCPM-V-2.6 | Lightweight VLM with structured prompt, batched via LMDeploy TurboMind |
| **Few-Shot Learning (FSL)** | Qwen2-VL-7B | Dynamic exemplar selection per entity type and product category |
| **Supervised Fine-Tuning (SFT)** | Qwen2-VL-7B | 8-bit QLoRA via LLaMA-Factory on 150K samples |

### Entity Types

| Entity | Allowed Units |
|--------|--------------|
| `width`, `depth`, `height` | centimetre, foot, inch, metre, millimetre, yard |
| `item_weight`, `maximum_weight_recommendation` | gram, kilogram, microgram, milligram, ounce, pound, ton |
| `voltage` | kilovolt, millivolt, volt |
| `wattage` | kilowatt, watt |
| `item_volume` | centilitre, cubic foot, cubic inch, cup, decilitre, fluid ounce, gallon, imperial gallon, litre, microlitre, millilitre, pint, quart |

## Project Structure

```
Image-Entity-Extraction/
|
+-- main.py                          # CLI orchestrator for all pipeline stages
+-- Makefile                         # Make commands for each stage
+-- requirements.txt                 # Python dependencies
+-- setup.py                         # Package setup
|
+-- src/
|   +-- __init__.py
|   +-- config.py                    # Central configuration (paths, models, hyperparams)
|   +-- constants.py                 # Entity-unit mappings and allowed units
|   +-- utils.py                     # Image download, parsing, post-processing
|   +-- sanity.py                    # Output format validation and F1 computation
|   |
|   +-- data/
|   |   +-- make_dataset.py          # Image downloading, train/test prep, exemplar pools
|   |
|   +-- features/
|   |   +-- build_features.py        # Image metadata extraction, entity analysis
|   |
|   +-- models/
|   |   +-- minicpm_zsp.py           # MiniCPM-V-2.6 zero-shot inference
|   |   +-- qwen2_zsp.py             # Qwen2-VL-7B zero-shot inference
|   |   +-- qwen2_fsl.py             # Qwen2-VL-7B few-shot inference
|   |   +-- qwen2_sft.py             # Qwen2-VL-7B fine-tuned inference (QLoRA)
|   |   +-- ensemble.py              # Majority-vote ensemble combiner
|   |   +-- train_model.py           # QLoRA fine-tuning via LLaMA-Factory
|   |   +-- predict_model.py         # Prediction orchestrator
|   |
|   +-- visualization/
|   |   +-- visualize.py             # Distribution plots, coverage, F1 comparison
|   |
|   +-- scripts/
|       +-- notebooks/
|           +-- MiniCPM.ipynb         # Interactive MiniCPM notebook
|
+-- data/                            # Data directory (gitignored)
|   +-- raw/                         # Original train.csv, test.csv
|   +-- processed/                   # Processed CSVs, exemplar pools
|   +-- images/
|       +-- train/                   # Downloaded training images
|       +-- test/                    # Downloaded test images
|
+-- models/                          # Model checkpoints (gitignored)
+-- output/                          # Prediction CSVs (gitignored)
+-- reports/
    +-- figures/                     # Generated analysis plots
```

## Quick Start

### 1. Setup

```bash
git clone <repo-url>
cd Image-Entity-Extraction

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 2. Prepare Data

Place `train.csv` and `test.csv` in `data/raw/`, then:

```bash
python main.py --stage data
```

Downloads all product images and builds few-shot exemplar pools.

### 3. Train (Optional)

```bash
python main.py --stage train
```

Generates LLaMA-Factory config and launches QLoRA fine-tuning on Qwen2-VL-7B.

### 4. Predict

```bash
# Full ensemble (MiniCPM ZSP + Qwen2 FSL + Qwen2 SFT)
python main.py --stage predict --model ensemble

# Single model
python main.py --stage predict --model minicpm_zsp
python main.py --stage predict --model qwen2_fsl
python main.py --stage predict --model qwen2_sft
```

### 5. Validate

```bash
python main.py --stage validate
```

### 6. Full Pipeline

```bash
python main.py --stage all
# or
make all
```

## Pipeline Stages

| Stage | Command | Description |
|-------|---------|-------------|
| Data Prep | `make data` | Download images, build exemplar pools |
| Features | `make features` | Extract image metadata, analyze distributions |
| Training | `make train` | QLoRA fine-tuning via LLaMA-Factory |
| Prediction | `make predict` | Ensemble inference across all models |
| Validation | `make validate` | Check output format against competition spec |
| Visualization | `make visualize` | Generate analysis plots |

## Hardware Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM (A100/A6000 recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: ~50GB for images + model weights

## Evaluation

F1 score via exact string matching:

| Condition | Classification |
|-----------|---------------|
| Prediction and ground truth both non-empty and equal | True Positive |
| Prediction non-empty but doesn't match ground truth | False Positive |
| Prediction empty but ground truth non-empty | False Negative |
| Both empty | True Negative |

```
F1 = 2 * Precision * Recall / (Precision + Recall)
```

## Author

Mukul Bhele
