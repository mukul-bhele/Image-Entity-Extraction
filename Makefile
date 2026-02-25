.PHONY: clean data features train predict predict-minicpm predict-qwen2-fsl validate visualize all requirements lint help

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = Image Entity Extraction
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Run the full end-to-end pipeline
all:
	$(PYTHON_INTERPRETER) main.py --stage all

## Download images and prepare data
data:
	$(PYTHON_INTERPRETER) main.py --stage data

## Run feature analysis
features:
	$(PYTHON_INTERPRETER) main.py --stage features

## Fine-tune Qwen2-VL with QLoRA
train:
	$(PYTHON_INTERPRETER) main.py --stage train

## Run prediction with ensemble (all models)
predict:
	$(PYTHON_INTERPRETER) main.py --stage predict --model ensemble

## Run prediction with MiniCPM zero-shot only
predict-minicpm:
	$(PYTHON_INTERPRETER) main.py --stage predict --model minicpm_zsp

## Run prediction with Qwen2 few-shot only
predict-qwen2-fsl:
	$(PYTHON_INTERPRETER) main.py --stage predict --model qwen2_fsl

## Validate output format
validate:
	$(PYTHON_INTERPRETER) main.py --stage validate

## Generate visualization plots
visualize:
	$(PYTHON_INTERPRETER) main.py --stage visualize

## Install Python dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make all               - Run the full end-to-end pipeline"
	@echo "  make data              - Download images and prepare data"
	@echo "  make features          - Run feature analysis"
	@echo "  make train             - Fine-tune Qwen2-VL with QLoRA"
	@echo "  make predict           - Run ensemble prediction"
	@echo "  make predict-minicpm   - Run MiniCPM zero-shot only"
	@echo "  make predict-qwen2-fsl - Run Qwen2 few-shot only"
	@echo "  make validate          - Validate output format"
	@echo "  make visualize         - Generate visualization plots"
	@echo "  make requirements      - Install dependencies"
	@echo "  make clean             - Remove compiled Python files"
	@echo "  make lint              - Lint source code"
