"""Анализ разметки VisDrone и сохранение метрик и графиков в MLflow."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import mlflow

# Скрипт запускается из корня проекта, где находится локальная копия датасета.
DATASET_DIR = Path("data/visdrone")
ANNOTATIONS_DIR = DATASET_DIR / "annotations"

# Номера классов соответствуют категориям в исходной разметке VisDrone.
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

# Сортировка сохраняет порядок соответствия аннотаций и статистики изображений.
class_counts = Counter()
annotations = sorted(ANNOTATIONS_DIR.glob("*.txt"))

if not annotations:
    raise FileNotFoundError(f"No annotations found in {ANNOTATIONS_DIR}")

# Накапливаем площади рамок, число объектов и число транспортных средств.
relative_areas = []
objects_per_image = []
invalid_boxes = 0
VEHICLE_CLASSES = {4, 5, 6, 7, 8, 9, 10}
vehicles_per_image = []

for annotation_path in annotations:
    vehicle_count = 0
    image_path = DATASET_DIR / "images" / f"{annotation_path.stem}.jpg"

    # Размер изображения нужен для перевода площади рамки в проценты кадра.
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    objects_count = 0

    for line in annotation_path.read_text().splitlines():
        if not line.strip():
            continue

        # Строка содержит координаты, размеры, score, класс, усечение и перекрытие.
        values = line.split(",")

        if len(values) != 8:
            raise ValueError(f"Invalid annotation in {annotation_path}")

        x, y, w, h, score, category, truncation, occlusion = map(int, values)

        # Пропускаем игнорируемые объекты, неизвестные классы и вырожденные рамки.
        if score == 0 or category not in CLASS_NAMES:
            continue

        if w <= 0 or h <= 0:
            invalid_boxes += 1
            continue

        # Считаем классы и относительные площади только по допустимой разметке.
        class_counts[category] += 1

        relative_area = (w * h) / (image_width * image_height) * 100

        relative_areas.append(relative_area)
        objects_count += 1

        if category in VEHICLE_CLASSES:
            vehicle_count += 1

        relative_area = (w * h) / (image_width * image_height) * 100

        relative_areas.append(relative_area)
        objects_count += 1

        if category in VEHICLE_CLASSES:
            vehicle_count += 1

    # Каждый элемент списков соответствует одному файлу из annotations.
    objects_per_image.append(objects_count)
    vehicles_per_image.append(vehicle_count)

# Выбираем кадры с минимальным, ближайшим к медиане и максимальным числом машин.
median_count = float(np.median(vehicles_per_image))

selected_indices = [
    int(np.argmin(vehicles_per_image)),
    min(
        range(len(vehicles_per_image)),
        key=lambda i: abs(vehicles_per_image[i] - median_count),
    ),
    int(np.argmax(vehicles_per_image)),
]

# Выводим краткую сводку, чтобы проверить прочитанную выборку в терминале.
print("Annotation files:", len(annotations))
print("Total objects:", sum(class_counts.values()))

for category, count in sorted(class_counts.items()):
    print(CLASS_NAMES[category], count)

# Показываем все классы, в том числе отсутствующие в выбранной части датасета.
names = [CLASS_NAMES[i] for i in CLASS_NAMES]
counts = [class_counts.get(i, 0) for i in CLASS_NAMES]

# Строим распределение объектов по классам.
fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(names, counts)

ax.set_title("VisDrone: Class Distribution")
ax.set_xlabel("Object class")
ax.set_ylabel("Number of objects")

plt.xticks(rotation=45, ha="right")
fig.tight_layout()

# Все результаты EDA сохраняются в один эксперимент локального MLflow.
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("visdrone-eda")

# Логарифмическая шкала позволяет сравнить маленькие и большие рамки.
fig2, ax2 = plt.subplots(figsize=(11, 6))

areas = np.array(relative_areas)

bins = np.geomspace(areas.min(), areas.max(), 50)

ax2.hist(areas, bins=bins)
ax2.set_xscale("log")

ax2.set_title("VisDrone: Relative Bounding Box Area")
ax2.set_xlabel("Bounding box area (% of image, log scale)")
ax2.set_ylabel("Number of objects")

fig2.tight_layout()

# Распределение числа транспортных средств показывает различия между кадрами.
fig3, ax3 = plt.subplots(figsize=(10, 6))

ax3.hist(vehicles_per_image, bins=30)

ax3.set_title("VisDrone: Vehicles per Image")
ax3.set_xlabel("Number of vehicles")
ax3.set_ylabel("Number of images")

fig3.tight_layout()


def draw_vehicle_boxes(annotation_path: Path) -> Image.Image:
    """Нарисовать допустимые рамки транспортных средств на копии изображения."""
    image_path = DATASET_DIR / "images" / f"{annotation_path.stem}.jpg"

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for line in annotation_path.read_text().splitlines():
        if not line.strip():
            continue

        x, y, w, h, score, category, *_ = map(int, line.split(","))

        # Для примеров плотности оставляем только транспорт с допустимой рамкой.
        if score == 0 or category not in VEHICLE_CLASSES:
            continue

        if w <= 0 or h <= 0:
            continue

        draw.rectangle(
            (x, y, x + w, y + h),
            outline="red",
            width=2,
        )

    return image


# Сравниваем три выбранных кадра вместе с исходными рамками транспорта.
fig4, axes = plt.subplots(1, 3, figsize=(18, 6))

labels = ["Minimum", "Median", "Maximum"]

for ax, idx, label in zip(axes, selected_indices, labels):
    image = draw_vehicle_boxes(annotations[idx])

    ax.imshow(image)
    ax.set_title(f"{label}: {vehicles_per_image[idx]} vehicles")
    ax.axis("off")

    image.close()

fig4.tight_layout()

# Один run объединяет численные метрики и четыре графика текущего запуска.
with mlflow.start_run(run_name="validation-eda-v2"):
    mlflow.log_param("dataset", "VisDrone2019-DET-val")
    mlflow.log_metric("images_count", len(annotations))
    mlflow.log_metric("objects_count", sum(class_counts.values()))

    mlflow.log_figure(fig, "plots/class_distribution.png")

    mlflow.log_figure(fig2, "plots/bbox_area_distribution.png")

    mlflow.log_metric("median_bbox_area_pct", float(np.median(areas)))

    mlflow.log_metric(
        "mean_objects_per_image",
        float(np.mean(objects_per_image)),
    )

    mlflow.log_metric("invalid_boxes", invalid_boxes)

    mlflow.log_figure(
        fig3,
        "plots/vehicles_per_image.png",
    )

    mlflow.log_metric(
        "mean_vehicles_per_image",
        float(np.mean(vehicles_per_image)),
    )

    mlflow.log_metric(
        "median_vehicles_per_image",
        float(np.median(vehicles_per_image)),
    )

    mlflow.log_metric(
        "max_vehicles_per_image",
        int(np.max(vehicles_per_image)),
    )

    mlflow.log_figure(
        fig4,
        "plots/vehicle_density_examples.png",
    )

# Освобождаем фигуры после загрузки артефактов в MLflow.
plt.close(fig)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)

print("EDA results saved to MLflow")
