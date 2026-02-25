# -*- coding: utf-8 -*-
"""
Sanity checker for the output CSV.
Validates format compliance against the competition requirements.
"""

import re
import sys
import pandas as pd
from src.constants import ENTITY_UNIT_MAP, ALLOWED_UNITS


def check_output_format(output_csv: str, test_csv: str) -> bool:
    """
    Validate that the output CSV meets all competition format requirements.

    Checks:
    1. Correct columns (index, prediction)
    2. All test indices are present
    3. Each prediction is either empty or in "value unit" format
    4. Values are valid floats (no scientific notation)
    5. Units are from the allowed set

    Returns:
        True if all checks pass, False otherwise
    """
    print(f"Validating output: {output_csv}")
    print(f"Against test file: {test_csv}")

    try:
        output_df = pd.read_csv(output_csv)
        test_df = pd.read_csv(test_csv)
    except Exception as e:
        print(f"FAIL: Could not read CSV files: {e}")
        return False

    # Check columns
    required_cols = {"index", "prediction"}
    if not required_cols.issubset(set(output_df.columns)):
        print(f"FAIL: Missing columns. Required: {required_cols}, Found: {set(output_df.columns)}")
        return False

    # Check row count
    if len(output_df) != len(test_df):
        print(f"FAIL: Row count mismatch. Output: {len(output_df)}, Test: {len(test_df)}")
        return False

    # Check all indices present
    test_indices = set(test_df["index"].tolist())
    output_indices = set(output_df["index"].tolist())
    missing = test_indices - output_indices
    if missing:
        print(f"FAIL: Missing {len(missing)} indices in output")
        return False

    extra = output_indices - test_indices
    if extra:
        print(f"FAIL: {len(extra)} extra indices in output not in test")
        return False

    # Validate each prediction
    errors = []
    # Merge to get entity_name for unit validation
    merged = output_df.merge(test_df[["index", "entity_name"]], on="index", how="left")

    for idx, row in merged.iterrows():
        prediction = str(row["prediction"]) if pd.notna(row["prediction"]) else ""
        entity_name = row.get("entity_name", "")

        if prediction.strip() == "" or prediction == "nan":
            continue  # Empty predictions are valid

        # Check for scientific notation
        if re.search(r"[eE][+-]?\d+", prediction):
            errors.append(f"Index {row['index']}: Scientific notation not allowed: '{prediction}'")
            continue

        # Check format: "value unit"
        parts = prediction.strip().split(" ", 1)
        if len(parts) != 2:
            errors.append(f"Index {row['index']}: Invalid format (need 'value unit'): '{prediction}'")
            continue

        value_str, unit = parts

        # Validate value is a valid float
        try:
            float(value_str)
        except ValueError:
            errors.append(f"Index {row['index']}: Invalid value '{value_str}' in '{prediction}'")
            continue

        # Validate unit
        entity_units = ENTITY_UNIT_MAP.get(entity_name, ALLOWED_UNITS)
        if unit not in entity_units:
            errors.append(f"Index {row['index']}: Invalid unit '{unit}' for entity '{entity_name}'")

    if errors:
        print(f"FAIL: {len(errors)} validation errors found:")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        return False

    print(f"PASS: All {len(output_df)} predictions are valid.")
    return True


def compute_f1(output_csv: str, ground_truth_csv: str) -> dict:
    """
    Compute F1 score using the competition's exact-match evaluation.

    Returns:
        Dict with tp, fp, fn, precision, recall, f1
    """
    output_df = pd.read_csv(output_csv)
    gt_df = pd.read_csv(ground_truth_csv)

    merged = output_df.merge(gt_df, on="index", suffixes=("_out", "_gt"))

    tp = fp = fn = 0

    for _, row in merged.iterrows():
        out = str(row.get("prediction_out", "")).strip() if pd.notna(row.get("prediction_out")) else ""
        gt = str(row.get("entity_value", row.get("prediction_gt", ""))).strip()
        if pd.isna(gt) or gt == "nan":
            gt = ""

        if out != "" and gt != "":
            if out == gt:
                tp += 1
            else:
                fp += 1
        elif out != "" and gt == "":
            fp += 1
        elif out == "" and gt != "":
            fn += 1
        # out == "" and gt == "" -> True Negative (not counted for F1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.sanity <output.csv> <test.csv>")
        sys.exit(1)

    output_file = sys.argv[1]
    test_file = sys.argv[2]

    passed = check_output_format(output_file, test_file)
    sys.exit(0 if passed else 1)
