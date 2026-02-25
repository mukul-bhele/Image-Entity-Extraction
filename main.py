# -*- coding: utf-8 -*-
"""
Main entry point for the Image Entity Extraction pipeline.

End-to-end workflow:
    1. Data Preparation   - Download images, build few-shot exemplar pools
    2. Feature Analysis   - Extract metadata, analyze distributions
    3. Training (optional)- QLoRA fine-tune Qwen2-VL-7B via LLaMA-Factory
    4. Inference          - Run MiniCPM ZSP, Qwen2 FSL, Qwen2 SFT
    5. Ensemble           - Majority-vote across model predictions
    6. Post-Processing    - Normalize units, handle fractions/ranges
    7. Validation         - Sanity-check output format
    8. Visualization      - Generate analysis plots

Usage:
    python main.py --stage all
    python main.py --stage data
    python main.py --stage train
    python main.py --stage predict --model minicpm_zsp
    python main.py --stage predict --model ensemble
    python main.py --stage validate
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def stage_data(args):
    """Stage 1: Data preparation."""
    from src.data.make_dataset import make_dataset

    logger.info("=" * 60)
    logger.info("STAGE 1: DATA PREPARATION")
    logger.info("=" * 60)

    train_df, test_df = make_dataset(
        download_images_flag=not args.skip_download,
        max_workers=args.workers,
    )
    logger.info(f"Training samples: {len(train_df)}")
    if test_df is not None:
        logger.info(f"Test samples: {len(test_df)}")

    return train_df, test_df


def stage_features(args):
    """Stage 2: Feature analysis."""
    import pandas as pd
    from src.config import PROCESSED_DATA_DIR, TRAIN_IMAGE_DIR
    from src.features.build_features import build_features, analyze_entity_distribution
    from src.visualization.visualize import plot_entity_distribution

    logger.info("=" * 60)
    logger.info("STAGE 2: FEATURE ANALYSIS")
    logger.info("=" * 60)

    train_path = PROCESSED_DATA_DIR / "train_processed.csv"
    if not train_path.exists():
        logger.error("Run data preparation first: python main.py --stage data")
        return

    train_df = pd.read_csv(train_path)
    enriched_df = build_features(train_df, image_dir=str(TRAIN_IMAGE_DIR))

    distribution = analyze_entity_distribution(enriched_df)
    for entity, info in distribution.items():
        logger.info(f"  {entity}: {info['count']} ({info['percentage']}%)")

    plot_entity_distribution(enriched_df)
    logger.info("Feature analysis complete.")


def stage_train(args):
    """Stage 3: Model training (QLoRA fine-tuning)."""
    from src.models.train_model import train

    logger.info("=" * 60)
    logger.info("STAGE 3: MODEL TRAINING (QLoRA)")
    logger.info("=" * 60)

    train()
    logger.info("Training stage complete.")


def stage_predict(args):
    """Stage 4-6: Inference + Ensemble + Post-processing."""
    import pandas as pd
    from src.models.predict_model import (
        load_test_data, predict_single_model,
        predict_ensemble, generate_final_output,
    )
    from src.config import TEST_CSV

    logger.info("=" * 60)
    logger.info("STAGE 4-6: PREDICTION + ENSEMBLE + POST-PROCESSING")
    logger.info("=" * 60)

    test_df = load_test_data()
    logger.info(f"Loaded {len(test_df)} test samples")

    model = args.model or "ensemble"

    if model == "ensemble":
        models_to_use = args.ensemble_models.split(",") if args.ensemble_models else None
        result_df = predict_ensemble(test_df, models=models_to_use)
    else:
        result_df = predict_single_model(model, test_df)

    test_csv_path = str(TEST_CSV) if TEST_CSV.exists() else None
    generate_final_output(result_df, test_csv_path=test_csv_path)

    logger.info("Prediction stage complete.")


def stage_validate(args):
    """Stage 7: Output validation."""
    from src.sanity import check_output_format, compute_f1
    from src.config import FINAL_OUTPUT_CSV, TEST_CSV, TRAIN_CSV

    logger.info("=" * 60)
    logger.info("STAGE 7: VALIDATION")
    logger.info("=" * 60)

    output_path = args.output_csv or str(FINAL_OUTPUT_CSV)
    test_path = args.test_csv or str(TEST_CSV)

    if not Path(output_path).exists():
        logger.error(f"Output file not found: {output_path}")
        return

    passed = check_output_format(output_path, test_path)

    # Compute F1 if ground truth is available (e.g., validation split)
    if args.ground_truth:
        scores = compute_f1(output_path, args.ground_truth)
        logger.info(f"F1 Score: {scores['f1']}")
        logger.info(f"Precision: {scores['precision']} | Recall: {scores['recall']}")
        logger.info(f"TP: {scores['tp']} | FP: {scores['fp']} | FN: {scores['fn']}")


def stage_visualize(args):
    """Stage 8: Generate visualizations."""
    import pandas as pd
    from src.config import FINAL_OUTPUT_CSV, PROCESSED_DATA_DIR
    from src.visualization.visualize import (
        plot_entity_distribution, plot_prediction_coverage, plot_unit_distribution,
    )

    logger.info("=" * 60)
    logger.info("STAGE 8: VISUALIZATION")
    logger.info("=" * 60)

    train_path = PROCESSED_DATA_DIR / "train_processed.csv"
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        plot_entity_distribution(train_df)

    if Path(str(FINAL_OUTPUT_CSV)).exists():
        pred_df = pd.read_csv(FINAL_OUTPUT_CSV)
        plot_prediction_coverage(pred_df)
        plot_unit_distribution(pred_df)

    logger.info("Visualization complete.")


def run_all(args):
    """Run the complete end-to-end pipeline."""
    stage_data(args)
    stage_features(args)
    if not args.skip_training:
        stage_train(args)
    stage_predict(args)
    stage_validate(args)
    stage_visualize(args)


def main():
    parser = argparse.ArgumentParser(
        description="Image Entity Extraction - Amazon ML Challenge 2024",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage all                         # Run entire pipeline
  python main.py --stage data                        # Download images & prepare data
  python main.py --stage data --skip-download        # Prepare data without downloading
  python main.py --stage train                       # Fine-tune Qwen2-VL with QLoRA
  python main.py --stage predict --model minicpm_zsp # Run MiniCPM zero-shot only
  python main.py --stage predict --model qwen2_fsl   # Run Qwen2 few-shot only
  python main.py --stage predict --model ensemble    # Run full ensemble
  python main.py --stage validate                    # Validate output format
  python main.py --stage visualize                   # Generate analysis plots
        """,
    )

    parser.add_argument(
        "--stage",
        choices=["all", "data", "features", "train", "predict", "validate", "visualize"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--model",
        choices=["minicpm_zsp", "qwen2_zsp", "qwen2_fsl", "qwen2_sft", "ensemble"],
        default=None,
        help="Model to use for prediction (default: ensemble)",
    )
    parser.add_argument(
        "--ensemble-models",
        type=str,
        default=None,
        help="Comma-separated model names for ensemble (e.g., 'minicpm_zsp,qwen2_fsl,qwen2_sft')",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip image downloading (use already downloaded images)",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training stage in 'all' mode",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers for image downloading",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Path to output CSV for validation",
    )
    parser.add_argument(
        "--test-csv", type=str, default=None,
        help="Path to test CSV for validation",
    )
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="Path to ground truth CSV for F1 computation",
    )

    args = parser.parse_args()

    stage_map = {
        "all": run_all,
        "data": stage_data,
        "features": stage_features,
        "train": stage_train,
        "predict": stage_predict,
        "validate": stage_validate,
        "visualize": stage_visualize,
    }

    stage_fn = stage_map[args.stage]
    stage_fn(args)


if __name__ == "__main__":
    main()
