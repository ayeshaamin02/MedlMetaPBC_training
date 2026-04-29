"""
Create_XAI_Combined_Figure.py

Creates one clean combined Grad-CAM figure from:
    xai_results/GradCAM_BestRuns/

Output:
    xai_results/GradCAM_BestRuns/GradCAM_BestRuns_Combined.png
"""

from pathlib import Path
import re
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("xai_results/GradCAM_BestRuns")
OUT = ROOT / "GradCAM_BestRuns_Combined_Edited.png"
FAMILIES = ["ConvNeXtBase", "EfficientNetV2S", "SwinBase"]
COLUMNS = [
    ("Original image", "original_image.png"),
    ("Grad-CAM heatmap", "heatmap.png"),
    ("Overlay", "gradcam_overlay.png"),
]


def font(path: str, size: int, fallback: ImageFont.ImageFont) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return fallback


def load_fonts() -> dict[str, ImageFont.ImageFont]:
    fallback = ImageFont.load_default()
    base = "/System/Library/Fonts/Supplemental"
    return {
        "title": font(f"{base}/Arial Bold.ttf", 28, fallback),
        "header": font(f"{base}/Arial Bold.ttf", 18, fallback),
        "label": font(f"{base}/Arial Bold.ttf", 15, fallback),
        "body": font(f"{base}/Arial.ttf", 14, fallback),
        "small": font(f"{base}/Arial.ttf", 12, fallback),
    }


def paste_centered(canvas: Image.Image, image_path: Path, box: tuple[int, int, int, int]) -> None:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((box[2] - box[0], box[3] - box[1]))
    x = box[0] + ((box[2] - box[0]) - img.width) // 2
    y = box[1] + ((box[3] - box[1]) - img.height) // 2
    canvas.paste(img, (x, y))


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width: int,
    line_spacing: int = 4,
) -> int:
    """Draw wrapped text and return the y position after the final line."""
    x, y = xy
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_spacing
    return y


def compact_run_label(value: object) -> str:
    text = str(value)
    round_match = re.search(r"Round\s*[_-]?(\d+)", text, flags=re.IGNORECASE)
    seed_match = re.search(r"seed\s*[_-]?(\d+)", text, flags=re.IGNORECASE)

    parts = []
    if round_match:
        parts.append(f"Round {round_match.group(1)}")
    else:
        parts.append(textwrap.shorten(text, width=28, placeholder="..."))

    if seed_match:
        parts.append(f"seed {seed_match.group(1)}")
    return ", ".join(parts)


def main() -> None:
    summary_path = ROOT / "xai_prediction_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}. Run Run_XAI_GradCAM_BestRuns.py first.")

    summary = pd.read_csv(summary_path)
    fonts = load_fonts()

    left_w = 330
    cell_w = 285
    row_h = 330
    top_h = 115
    footer_h = 45
    pad = 28
    img_box = 224

    width = pad * 2 + left_w + len(COLUMNS) * cell_w
    height = top_h + len(FAMILIES) * row_h + footer_h

    blue = (31, 78, 121)
    light_blue = (234, 242, 248)
    grid = (215, 225, 235)
    dark = (30, 30, 30)
    muted = (90, 90, 90)
    green = (34, 118, 75)

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((pad, 24), "Grad-CAM XAI Results: Best Run Per Model", fill=blue, font=fonts["title"])
    draw.text(
        (pad, 62),
        "Peripheral blood cell classification using one representative correctly classified test image per model",
        fill=muted,
        font=fonts["body"],
    )

    header_y = top_h - 36
    for j, (title, _) in enumerate(COLUMNS):
        x = pad + left_w + j * cell_w
        draw.text((x + 55, header_y), title, fill=blue, font=fonts["header"])

    for i, family in enumerate(FAMILIES):
        row = summary[summary["model_family"] == family]
        if row.empty:
            continue
        row = row.iloc[0]
        y = top_h + i * row_h

        if i % 2 == 0:
            draw.rectangle([0, y, width, y + row_h], fill=(250, 252, 254))
        draw.line([pad, y, width - pad, y], fill=grid, width=1)

        label_x = pad
        label_y = y + 52
        draw.text((label_x, label_y), family, fill=blue, font=fonts["header"])
        run_y = draw_wrapped(
            draw,
            (label_x, label_y + 34),
            f"Run: {compact_run_label(row.get('round', ''))}",
            fonts["small"],
            dark,
            width=left_w - 30,
        )
        draw.text((label_x, run_y + 4), f"Macro-F1: {float(row['selection_metric_value']):.4f}", fill=dark, font=fonts["small"])
        draw.text((label_x, run_y + 38), f"True: {row['true_class']}", fill=green, font=fonts["label"])
        draw.text((label_x, run_y + 60), f"Predicted: {row['predicted_class']}", fill=green, font=fonts["label"])
        draw.text((label_x, run_y + 82), f"Confidence: {float(row['confidence']):.3f}", fill=green, font=fonts["label"])

        for j, (_, filename) in enumerate(COLUMNS):
            x = pad + left_w + j * cell_w
            img_y = y + 50
            box = (x + 28, img_y, x + 28 + img_box, img_y + img_box)
            draw.rounded_rectangle(
                [box[0] - 8, box[1] - 8, box[2] + 8, box[3] + 8],
                radius=8,
                fill="white",
                outline=grid,
                width=1,
            )
            paste_centered(canvas, ROOT / family / filename, box)

    draw.line([pad, height - footer_h, width - pad, height - footer_h], fill=grid, width=1)
    draw.text(
        (pad, height - 30),
        "Grad-CAM overlays are qualitative explanations of the predicted class, not statistical tests.",
        fill=muted,
        font=fonts["small"],
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(OUT.resolve())


if __name__ == "__main__":
    main()
