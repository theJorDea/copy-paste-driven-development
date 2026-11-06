
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = [
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

SPLITS = {
    "training": "visdrone-train",
    "validation": "visdrone",
}


def draw_labels(image_path: Path, label_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size

    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id, xc, yc, w, h = line.split()

        class_id = int(class_id)
        xc, yc, w, h = map(float, (xc, yc, w, h))

        x1 = int((xc - w / 2) * image_width)
        y1 = int((yc - h / 2) * image_height)
        x2 = int((xc + w / 2) * image_width)
        y2 = int((yc + h / 2) * image_height)

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=2,
        )

        draw.text(
            (x1, max(0, y1 - 12)),
            CLASS_NAMES[class_id],
            fill="red",
        )

    return image


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=SPLITS.keys(),
        required=True,
    )

    parser.add_argument("--count", type=int, default=6)

    args = parser.parse_args()

    dataset_dir = ROOT / "data" / SPLITS[args.split]
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    images = sorted(images_dir.glob("*.jpg"))

    random.seed(42)
    selected = random.sample(images, min(args.count, len(images)))

    output_dir = ROOT / "outputs" / f"preview-{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in selected:
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.is_file():
            raise FileNotFoundError(label_path)

        image = draw_labels(image_path, label_path)

        output_path = output_dir / image_path.name
        image.save(output_path)
        image.close()

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()