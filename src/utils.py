# -*- coding: utf-8 -*-
"""
Utility functions for the Image Entity Extraction pipeline.
Includes image downloading, sanity checking, and parsing helpers.
"""

import os
import re
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.constants import ENTITY_UNIT_MAP, ALLOWED_UNITS


# ──────────────────────────────────────────────────────────────────────
# Image downloading
# ──────────────────────────────────────────────────────────────────────

def download_image(image_url: str, save_dir: str, timeout: int = 15) -> Optional[str]:
    """Download a single image from URL to save_dir. Returns saved path or None."""
    try:
        filename = os.path.basename(urlparse(image_url).path)
        save_path = os.path.join(save_dir, filename)
        if os.path.exists(save_path):
            return save_path
        response = requests.get(image_url, timeout=timeout, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path
    except Exception:
        return None


def download_images(
    image_urls: list,
    save_dir: str,
    max_workers: int = 8,
    show_progress: bool = True,
) -> dict:
    """
    Download images in parallel.
    Returns dict mapping image_url -> local_path (or None if failed).
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(download_image, url, save_dir): url
            for url in image_urls
        }
        iterator = as_completed(future_to_url)
        if show_progress:
            iterator = tqdm(iterator, total=len(image_urls), desc="Downloading images")

        for future in iterator:
            url = future_to_url[future]
            results[url] = future.result()

    return results


def get_local_image_path(image_url: str, image_dir: str) -> str:
    """Convert an image URL to a local file path."""
    filename = image_url.split("/")[-1]
    return os.path.join(image_dir, filename)


# ──────────────────────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────────────────────

def parse_model_response(response_text: str, entity_name: str) -> str:
    """
    Parse the model's raw text response into 'value unit' format.
    Returns empty string if parsing fails.
    """
    if not response_text or response_text.strip().lower() == "none":
        return ""

    allowed_units = ENTITY_UNIT_MAP.get(entity_name, set())

    # Try structured Value:/Unit: format first
    value_match = re.search(r"Value:\s*([\d.,]+)", response_text)
    unit_match = re.search(r"Unit:\s*([a-zA-Z\s]+)", response_text)

    if value_match and unit_match:
        value = clean_numeric_value(value_match.group(1))
        unit = unit_match.group(1).strip().lower()
        unit = normalize_unit(unit, allowed_units)
        if value and unit:
            return f"{value} {unit}"

    # Fallback: try to extract number + unit from raw text
    pattern = r"([\d.,]+)\s*([a-zA-Z\s]+)"
    matches = re.findall(pattern, response_text)
    for val_str, unit_str in matches:
        value = clean_numeric_value(val_str)
        unit = normalize_unit(unit_str.strip().lower(), allowed_units)
        if value and unit:
            return f"{value} {unit}"

    return ""


def clean_numeric_value(value_str: str) -> str:
    """Clean and validate a numeric value string. Returns formatted float string."""
    try:
        value_str = value_str.replace(",", "")
        value = float(value_str)
        if value == int(value):
            return str(int(value))
        return str(value)
    except (ValueError, TypeError):
        return ""


def normalize_unit(unit_str: str, allowed_units: set) -> str:
    """Normalize a unit string to match an allowed unit. Returns empty string if no match."""
    unit_str = unit_str.strip().lower()

    # Direct match
    if unit_str in allowed_units:
        return unit_str

    # Common abbreviations / misspellings map
    unit_aliases = {
        "cm": "centimetre", "centimeter": "centimetre", "centimeters": "centimetre",
        "centimetres": "centimetre",
        "m": "metre", "meter": "metre", "meters": "metre", "metres": "metre",
        "mm": "millimetre", "millimeter": "millimetre", "millimeters": "millimetre",
        "millimetres": "millimetre",
        "ft": "foot", "feet": "foot",
        "in": "inch", "inches": "inch", "\"": "inch",
        "yd": "yard", "yards": "yard",
        "g": "gram", "grams": "gram", "gm": "gram", "gms": "gram",
        "kg": "kilogram", "kilograms": "kilogram", "kgs": "kilogram",
        "mg": "milligram", "milligrams": "milligram",
        "ug": "microgram", "micrograms": "microgram", "mcg": "microgram",
        "oz": "ounce", "ounces": "ounce",
        "lb": "pound", "lbs": "pound", "pounds": "pound",
        "tons": "ton", "tonnes": "ton",
        "v": "volt", "volts": "volt",
        "kv": "kilovolt", "kilovolts": "kilovolt",
        "mv": "millivolt", "millivolts": "millivolt",
        "w": "watt", "watts": "watt",
        "kw": "kilowatt", "kilowatts": "kilowatt",
        "l": "litre", "liter": "litre", "liters": "litre", "litres": "litre",
        "ml": "millilitre", "milliliter": "millilitre", "milliliters": "millilitre",
        "millilitres": "millilitre",
        "cl": "centilitre", "centiliter": "centilitre",
        "dl": "decilitre", "deciliter": "decilitre",
        "ul": "microlitre", "microliter": "microlitre",
        "fl oz": "fluid ounce", "fluid ounces": "fluid ounce",
        "gal": "gallon", "gallons": "gallon",
        "pt": "pint", "pints": "pint",
        "qt": "quart", "quarts": "quart",
        "cups": "cup",
        "cu ft": "cubic foot", "cubic feet": "cubic foot",
        "cu in": "cubic inch", "cubic inches": "cubic inch",
        "imp gal": "imperial gallon", "imperial gallons": "imperial gallon",
    }

    normalized = unit_aliases.get(unit_str, "")
    if normalized in allowed_units:
        return normalized

    # Partial match: check if any allowed unit is a substring
    for allowed in allowed_units:
        if allowed in unit_str or unit_str in allowed:
            return allowed

    return ""


# ──────────────────────────────────────────────────────────────────────
# Sanity checking / validation
# ──────────────────────────────────────────────────────────────────────

def validate_prediction(prediction: str, entity_name: str) -> bool:
    """Validate that a prediction matches the required format."""
    if prediction == "":
        return True  # empty predictions are valid (abstentions)

    pattern = r"^[\d.]+\s+\S+.*$"
    if not re.match(pattern, prediction):
        return False

    parts = prediction.split(" ", 1)
    if len(parts) != 2:
        return False

    value_str, unit = parts
    try:
        float(value_str)
    except ValueError:
        return False

    allowed = ENTITY_UNIT_MAP.get(entity_name, ALLOWED_UNITS)
    return unit in allowed


def sanity_check(output_csv: str, test_csv: str) -> Tuple[bool, str]:
    """
    Run sanity check on the output CSV against the test CSV.
    Returns (passed, message).
    """
    test_df = pd.read_csv(test_csv)
    output_df = pd.read_csv(output_csv)

    if len(output_df) != len(test_df):
        return False, f"Row count mismatch: output has {len(output_df)}, test has {len(test_df)}"

    if "index" not in output_df.columns or "prediction" not in output_df.columns:
        return False, "Output must have 'index' and 'prediction' columns"

    test_indices = set(test_df["index"].tolist())
    output_indices = set(output_df["index"].tolist())
    if test_indices != output_indices:
        missing = test_indices - output_indices
        return False, f"Missing indices in output: {len(missing)} entries"

    return True, "Sanity check passed"


# ──────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────

def handle_fractions(text: str) -> str:
    """Convert fraction characters and mixed fractions to decimal values."""
    fraction_map = {
        "½": ".5", "⅓": ".333", "⅔": ".667",
        "¼": ".25", "¾": ".75", "⅛": ".125",
        "⅜": ".375", "⅝": ".625", "⅞": ".875",
    }
    for frac_char, decimal in fraction_map.items():
        # Handle mixed fractions like "1½" -> "1.5"
        text = re.sub(rf"(\d+){frac_char}", lambda m: str(float(m.group(1)) + float(decimal)), text)
        # Handle standalone fractions like "½" -> "0.5"
        text = text.replace(frac_char, f"0{decimal}")
    return text


def handle_range(text: str) -> str:
    """When a range like '3-5' appears, select the higher value."""
    match = re.match(r"([\d.]+)\s*[-–]\s*([\d.]+)\s+(.*)", text)
    if match:
        val1, val2, unit = float(match.group(1)), float(match.group(2)), match.group(3)
        return f"{max(val1, val2)} {unit}"
    return text


def standardize_symbols(text: str) -> str:
    """Standardize quote characters for feet/inches measurements."""
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Convert feet/inches notation: 5'6" -> process as needed
    feet_inches = re.match(r"(\d+)['\u2032]\s*(\d+)?[\"″]?", text)
    if feet_inches:
        feet = float(feet_inches.group(1))
        inches = float(feet_inches.group(2) or 0)
        total_inches = feet * 12 + inches
        return f"{total_inches} inch"
    return text


def post_process_prediction(prediction: str, entity_name: str) -> str:
    """Apply all post-processing steps to a single prediction."""
    if not prediction or prediction.strip() == "":
        return ""

    prediction = handle_fractions(prediction)
    prediction = standardize_symbols(prediction)
    prediction = handle_range(prediction)

    # Re-parse after post-processing
    allowed_units = ENTITY_UNIT_MAP.get(entity_name, set())
    pattern = r"([\d.,]+)\s*([a-zA-Z\s]+)"
    match = re.search(pattern, prediction)
    if match:
        value = clean_numeric_value(match.group(1))
        unit = normalize_unit(match.group(2).strip(), allowed_units)
        if value and unit:
            return f"{value} {unit}"

    return ""
