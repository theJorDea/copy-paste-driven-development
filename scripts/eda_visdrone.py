from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow

DATASET_DIR = Path("data/visdrone")
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

CLASS_NAMES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

class_counts = Counter()
annotations = sorted(ANNOTATIONS_DIR.glob("*.txt"))

if not annotations:
    raise FileNotFoundError(
        f"No annotations found in {ANNOTATIONS_DIR}"
    )

for annotation_path in annotations:
    for line in annotation_path.read_text().splitlines():
        if not line.strip():
            continue

        values = line.split(",")

        if len(values) != 8:
            raise ValueError(f"Invalid annotation in {annotation_path}")

        x, y, w, h, score, category, truncation, occlusion = map(
            int, values
        )

        if score == 0 or category not in CLASS_NAMES:
            continue

        class_counts[category] += 1

print("Annotation files:", len(annotations))
print("Total objects:", sum(class_counts.values()))

for category, count in sorted(class_counts.items()):
    print(CLASS_NAMES[category], count)

# Формируем данные для графика
names = [CLASS_NAMES[i] for i in CLASS_NAMES]
counts = [class_counts.get(i, 0) for i in CLASS_NAMES]

# Создаём график
fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(names, counts)

ax.set_title("VisDrone: Class Distribution")
ax.set_xlabel("Object class")
ax.set_ylabel("Number of objects")

plt.xticks(rotation=45, ha="right")
fig.tight_layout()

# Подключаемся к MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("visdrone-eda")

# Создаём Run и сохраняем результаты
with mlflow.start_run(run_name="validation-class-distribution"):
    mlflow.log_param("dataset", "VisDrone2019-DET-val")
    mlflow.log_metric("images_count", len(annotations))
    mlflow.log_metric("objects_count", sum(class_counts.values()))

    mlflow.log_figure(fig, "plots/class_distribution.png")

plt.close(fig)

print("EDA results saved to MLflow")