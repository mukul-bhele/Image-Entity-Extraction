# -*- coding: utf-8 -*-
"""
Constants for the Image Entity Extraction pipeline.
Contains allowed units per entity type and entity-unit mappings.
"""

# Entity name to allowed units mapping
ENTITY_UNIT_MAP = {
    "width": {
        "centimetre", "foot", "inch", "metre", "millimetre", "yard"
    },
    "depth": {
        "centimetre", "foot", "inch", "metre", "millimetre", "yard"
    },
    "height": {
        "centimetre", "foot", "inch", "metre", "millimetre", "yard"
    },
    "item_weight": {
        "gram", "kilogram", "microgram", "milligram", "ounce", "pound", "ton"
    },
    "maximum_weight_recommendation": {
        "gram", "kilogram", "microgram", "milligram", "ounce", "pound", "ton"
    },
    "voltage": {
        "kilovolt", "millivolt", "volt"
    },
    "wattage": {
        "kilowatt", "watt"
    },
    "item_volume": {
        "centilitre", "cubic foot", "cubic inch", "cup", "decilitre",
        "fluid ounce", "gallon", "imperial gallon", "litre", "microlitre",
        "millilitre", "pint", "quart"
    },
}

# Flat set of all allowed units across all entity types
ALLOWED_UNITS = set()
for units in ENTITY_UNIT_MAP.values():
    ALLOWED_UNITS.update(units)

# Entity types that represent dimensions (length-based)
DIMENSION_ENTITIES = {"width", "depth", "height"}

# Entity types that represent weight
WEIGHT_ENTITIES = {"item_weight", "maximum_weight_recommendation"}

# Entity types that represent electrical properties
ELECTRICAL_ENTITIES = {"voltage", "wattage"}

# Entity types that represent volume
VOLUME_ENTITIES = {"item_volume"}

# All entity names
ALL_ENTITY_NAMES = set(ENTITY_UNIT_MAP.keys())
