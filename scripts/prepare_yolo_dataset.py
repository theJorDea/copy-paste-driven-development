import argparse
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

SPLITS = {
    "training": "visdrone-train",
    "validation": "visdrone",
}

CLASS_MAP = {
    4: 0,
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 5,
    10: 6,
}

CLASS_NAMES = [
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def convert_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(image_width, x + w)
    y2 = min(image_height, y + h)

    w_new = x2 - x1
    h_new = y2 - y1

    if w_new <= 0 or h_new <= 0:
        return None

    x_center = (x1 + x2) / (2 * image_width)
    y_center = (y1 + y2) / (2 * image_height)

    w_norm = w_new / image_width
    h_norm = h_new / image_height

    return x_center, y_center, w_norm, h_norm


def convert_annotation(
    annotation_path: Path,
    image_path: Path,
    label_path: Path,
) -> int:
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    label_lines = []

    for line_number, line in enumerate(
        annotation_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        raw_line = line.strip()

        if raw_line.endswith(","):
            raw_line = raw_line[:-1]

        parts = raw_line.split(",")

        if len(parts) != 8:
            raise ValueError(f"Invalid annotation at {annotation_path}:{line_number}")

        try:
            x, y, w, h, score, category, _, _ = map(int, parts)
        except ValueError as exc:
            raise ValueError(f"Invalid values at {annotation_path}:{line_number}") from exc

        if score == 0 or category not in CLASS_MAP:
            continue

        bbox = convert_bbox(x, y, w, h, image_width, image_height)

        if bbox is None:
            continue

        class_id = CLASS_MAP[category]

        label_lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in bbox))

    label_path.parent.mkdir(parents=True, exist_ok=True)

    label_path.write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""),
        encoding="utf-8",
    )

    return len(label_lines)


def process_dataset(dataset_dir: Path) -> dict[str, int]:
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    labels_dir = dataset_dir / "labels"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not annotations_dir.is_dir():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    image_paths = {
        path.stem: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    annotations = sorted(annotations_dir.glob("*.txt"))

    if not annotations:
        raise FileNotFoundError(f"No annotations found in {annotations_dir}")

    image_stems = set(image_paths)
    annotation_stems = {path.stem for path in annotations}

    missing_images = annotation_stems - image_stems
    missing_annotations = image_stems - annotation_stems

    if missing_images or missing_annotations:
        raise ValueError(
            f"Dataset mismatch: "
            f"{len(missing_images)} images missing, "
            f"{len(missing_annotations)} annotations missing"
        )

    total_objects = 0
    empty_labels = 0

    for annotation_path in annotations:
        image_path = image_paths[annotation_path.stem]
        label_path = labels_dir / f"{annotation_path.stem}.txt"

        objects_count = convert_annotation(
            annotation_path=annotation_path,
            image_path=image_path,
            label_path=label_path,
        )

        total_objects += objects_count

        if objects_count == 0:
            empty_labels += 1

    return {
        "images": len(annotations),
        "objects": total_objects,
        "empty_labels": empty_labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO format")

    parser.add_argument(
        "--split",
        choices=SPLITS.keys(),
        required=True,
    )

    args = parser.parse_args()

    dataset_dir = ROOT / "data" / SPLITS[args.split]

    print(f"Processing {args.split} dataset...")

    stats = process_dataset(dataset_dir)

    print("Dataset conversion completed!")
    print(f"Images: {stats['images']}")
    print(f"Objects: {stats['objects']}")
    print(f"Empty labels: {stats['empty_labels']}")


if __name__ == "__main__":
    main()
