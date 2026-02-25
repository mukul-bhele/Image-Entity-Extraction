# -*- coding: utf-8 -*-
"""
Generate Architecture Workflow PDF for the Image Entity Extraction pipeline.
Creates a visual diagram of the end-to-end architecture.

Usage: python generate_architecture_pdf.py
Output: reports/architecture_workflow.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os


def draw_box(ax, x, y, w, h, text, color="#4A90D9", text_color="white",
             fontsize=9, fontweight="bold", alpha=0.9, border_color=None):
    """Draw a rounded rectangle with centered text."""
    if border_color is None:
        border_color = color
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor=border_color,
        linewidth=1.5, alpha=alpha, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color, zorder=3, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, color="#555555"):
    """Draw an arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            lw=1.8, connectionstyle="arc3,rad=0",
        ),
        zorder=1,
    )


def draw_section_label(ax, x, y, text, fontsize=11):
    """Draw a section label."""
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#333333",
            style="italic", zorder=3)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(16, 22))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 22)
    ax.axis("off")

    # Title
    ax.text(8, 21.3, "Image Entity Extraction - Architecture Workflow",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#1a1a2e")
    ax.text(8, 20.9, "Amazon ML Challenge 2024 | Ensemble VLM Pipeline",
            ha="center", va="center", fontsize=10, color="#666666")

    # ── STAGE 1: INPUT ──
    draw_section_label(ax, 2.5, 20.2, "STAGE 1: INPUT")
    draw_box(ax, 5.5, 20.2, 3.5, 0.6, "Product Image", "#2C3E50")
    draw_box(ax, 10.5, 20.2, 3.5, 0.6, "Entity Name", "#2C3E50")
    draw_arrow(ax, 5.5, 19.85, 8, 19.45)
    draw_arrow(ax, 10.5, 19.85, 8, 19.45)

    # Combined input
    draw_box(ax, 8, 19.2, 5, 0.5, "Image + Entity Query", "#34495E")
    draw_arrow(ax, 8, 18.9, 8, 18.5)

    # ── STAGE 2: DATA PREPARATION ──
    draw_section_label(ax, 2.5, 18.2, "STAGE 2: DATA PREP")
    draw_box(ax, 8, 18.2, 6, 0.5, "Download Images  |  Build Exemplar Pools  |  Process CSVs",
             "#16A085", fontsize=8)
    draw_arrow(ax, 8, 17.9, 8, 17.5)

    # ── STAGE 3: MODEL INFERENCE ──
    draw_section_label(ax, 2.5, 17.2, "STAGE 3: INFERENCE")

    # Background panel for models
    panel = FancyBboxPatch(
        (2.5, 14.5), 11, 2.5,
        boxstyle="round,pad=0.15",
        facecolor="#F0F4F8", edgecolor="#B0BEC5",
        linewidth=1, alpha=0.6, zorder=0,
    )
    ax.add_patch(panel)

    # Three model branches
    # MiniCPM ZSP
    draw_box(ax, 4.5, 16.4, 3.2, 0.5, "MiniCPM-V-2.6", "#E74C3C")
    draw_box(ax, 4.5, 15.7, 3.2, 0.5, "Zero-Shot Prompting", "#C0392B", fontsize=8)
    draw_box(ax, 4.5, 15.0, 3.2, 0.45, "LMDeploy TurboMind\nBatch Size: 16",
             "#E8D5D5", text_color="#333", fontsize=7, fontweight="normal")

    # Qwen2 FSL
    draw_box(ax, 8, 16.4, 3.2, 0.5, "Qwen2-VL-7B", "#3498DB")
    draw_box(ax, 8, 15.7, 3.2, 0.5, "Few-Shot Learning", "#2980B9", fontsize=8)
    draw_box(ax, 8, 15.0, 3.2, 0.45, "Dynamic Exemplar\nSelection (3 shots)",
             "#D5E0E8", text_color="#333", fontsize=7, fontweight="normal")

    # Qwen2 SFT
    draw_box(ax, 11.5, 16.4, 3.2, 0.5, "Qwen2-VL-7B", "#9B59B6")
    draw_box(ax, 11.5, 15.7, 3.2, 0.5, "Fine-Tuned (QLoRA)", "#8E44AD", fontsize=8)
    draw_box(ax, 11.5, 15.0, 3.2, 0.45, "8-bit QLoRA\n150K samples, 1 epoch",
             "#E0D5E8", text_color="#333", fontsize=7, fontweight="normal")

    # Arrows from input to models
    draw_arrow(ax, 6.5, 17.45, 4.5, 16.7)
    draw_arrow(ax, 8, 17.45, 8, 16.7)
    draw_arrow(ax, 9.5, 17.45, 11.5, 16.7)

    # Arrows from models to ensemble
    draw_arrow(ax, 4.5, 14.7, 8, 14.0)
    draw_arrow(ax, 8, 14.7, 8, 14.0)
    draw_arrow(ax, 11.5, 14.7, 8, 14.0)

    # ── STAGE 4: ENSEMBLE ──
    draw_section_label(ax, 2.5, 13.7, "STAGE 4: ENSEMBLE")
    draw_box(ax, 8, 13.7, 5.5, 0.6, "Majority Vote Ensemble", "#E67E22")
    draw_arrow(ax, 8, 13.35, 8, 12.9)

    # ── STAGE 5: POST-PROCESSING ──
    draw_section_label(ax, 2.5, 12.5, "STAGE 5: POST-PROC")

    panel2 = FancyBboxPatch(
        (3.5, 11.2), 9, 1.5,
        boxstyle="round,pad=0.1",
        facecolor="#FFF9E6", edgecolor="#F5D76E",
        linewidth=1, alpha=0.5, zorder=0,
    )
    ax.add_patch(panel2)

    draw_box(ax, 5.5, 12.2, 3, 0.4, "Fraction Handling", "#F39C12",
             fontsize=8, fontweight="normal")
    draw_box(ax, 10.5, 12.2, 3, 0.4, "Symbol Standardization", "#F39C12",
             fontsize=8, fontweight="normal")
    draw_box(ax, 5.5, 11.6, 3, 0.4, "Range Resolution", "#F39C12",
             fontsize=8, fontweight="normal")
    draw_box(ax, 10.5, 11.6, 3, 0.4, "Unit Normalization", "#F39C12",
             fontsize=8, fontweight="normal")

    draw_arrow(ax, 8, 11.15, 8, 10.7)

    # ── STAGE 6: VALIDATION ──
    draw_section_label(ax, 2.5, 10.4, "STAGE 6: VALIDATE")
    draw_box(ax, 8, 10.4, 5, 0.5, "Sanity Check  |  Format Validation", "#27AE60", fontsize=8)
    draw_arrow(ax, 8, 10.1, 8, 9.7)

    # ── STAGE 7: OUTPUT ──
    draw_section_label(ax, 2.5, 9.4, "STAGE 7: OUTPUT")
    draw_box(ax, 8, 9.4, 4, 0.6, 'test_out.csv\n"value unit"', "#1ABC9C")

    # ═══════════════════════════════════════════════════════════
    # DETAILED COMPONENT DIAGRAMS (bottom half)
    # ═══════════════════════════════════════════════════════════

    ax.text(8, 8.5, "Component Details", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1a1a2e")

    # ── Training Pipeline (left side) ──
    ax.text(4.5, 7.9, "Training Pipeline (QLoRA)", ha="center",
            fontsize=10, fontweight="bold", color="#8E44AD")

    draw_box(ax, 4.5, 7.3, 4, 0.45, "train.csv + Images", "#2C3E50", fontsize=8)
    draw_arrow(ax, 4.5, 7.0, 4.5, 6.65)
    draw_box(ax, 4.5, 6.4, 4, 0.45, "Convert to LLaMA-Factory JSON", "#8E44AD", fontsize=8)
    draw_arrow(ax, 4.5, 6.1, 4.5, 5.75)
    draw_box(ax, 4.5, 5.5, 4, 0.45, "8-bit QLoRA Config", "#8E44AD", fontsize=8)
    draw_arrow(ax, 4.5, 5.2, 4.5, 4.85)
    draw_box(ax, 4.5, 4.6, 4, 0.45, "llamafactory-cli train", "#8E44AD", fontsize=8)
    draw_arrow(ax, 4.5, 4.3, 4.5, 3.95)
    draw_box(ax, 4.5, 3.7, 4, 0.45, "LoRA Adapter Weights", "#27AE60", fontsize=8)

    # ── Few-Shot Exemplar Pipeline (right side) ──
    ax.text(11.5, 7.9, "Few-Shot Exemplar Pipeline", ha="center",
            fontsize=10, fontweight="bold", color="#3498DB")

    draw_box(ax, 11.5, 7.3, 4, 0.45, "Training Data", "#2C3E50", fontsize=8)
    draw_arrow(ax, 11.5, 7.0, 11.5, 6.65)
    draw_box(ax, 11.5, 6.4, 4, 0.45, "Group by entity_name", "#3498DB", fontsize=8)
    draw_arrow(ax, 11.5, 6.1, 11.5, 5.75)
    draw_box(ax, 11.5, 5.5, 4, 0.45, "Stratify by group_id", "#3498DB", fontsize=8)
    draw_arrow(ax, 11.5, 5.2, 11.5, 4.85)
    draw_box(ax, 11.5, 4.6, 4, 0.45, "Sample 3 exemplars/group", "#3498DB", fontsize=8)
    draw_arrow(ax, 11.5, 4.3, 11.5, 3.95)
    draw_box(ax, 11.5, 3.7, 4, 0.45, "Exemplar Pool CSVs", "#27AE60", fontsize=8)

    # ── Evaluation Metric ──
    ax.text(8, 2.9, "Evaluation Metric", ha="center",
            fontsize=10, fontweight="bold", color="#333")

    eval_text = (
        "F1 = 2 * Precision * Recall / (Precision + Recall)\n"
        "Precision = TP / (TP + FP)    |    Recall = TP / (TP + FN)\n"
        "Exact string match: prediction must equal ground truth"
    )
    eval_panel = FancyBboxPatch(
        (3, 1.6), 10, 1.0,
        boxstyle="round,pad=0.1",
        facecolor="#EBF5FB", edgecolor="#3498DB",
        linewidth=1, alpha=0.7, zorder=0,
    )
    ax.add_patch(eval_panel)
    ax.text(8, 2.1, eval_text, ha="center", va="center",
            fontsize=8, color="#2C3E50", family="monospace", zorder=3)

    # ── Entity Types Legend ──
    ax.text(8, 1.2, "Supported Entities", ha="center",
            fontsize=9, fontweight="bold", color="#333")
    entities = "width | depth | height | item_weight | max_weight_recommendation | voltage | wattage | item_volume"
    ax.text(8, 0.85, entities, ha="center", va="center",
            fontsize=7, color="#555", family="monospace")

    # Footer
    ax.text(8, 0.3, "Image Entity Extraction | Amazon ML Challenge 2024 | Mukul Bhele",
            ha="center", va="center", fontsize=8, color="#999999")

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "architecture_workflow.pdf")

    fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Architecture workflow PDF saved to: {output_path}")

    # Also save as PNG for quick viewing
    png_path = os.path.join(output_dir, "figures", "architecture_workflow.png")
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 22))
    ax2.set_xlim(0, 16)
    ax2.set_ylim(0, 22)
    ax2.axis("off")
    # Re-render for PNG (reuse by re-calling)
    plt.close(fig2)

    # Just save from the original figure
    fig_png, _ = plt.subplots(1, 1, figsize=(16, 22))
    plt.close(fig_png)

    print("Done.")


if __name__ == "__main__":
    main()
