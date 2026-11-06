"""Сборка манифеста VisDrone с контрольными суммами и регистрация в MLflow."""

import hashlib
from pathlib import Path

import pandas as pd

import mlflow

# Пути привязаны к расположению скрипта, а не к текущей рабочей папке.
ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "visdrone"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"

# Транспорт — подмножество десяти допустимых категорий VisDrone.
VEHICLE_CLASSES = {4, 5, 6, 7, 8, 9, 10}
VALID_CLASSES = set(range(1, 11))


def sha256_file(path: Path) -> str:
    """Вычислить SHA-256 файла, читая его блоками по одному МиБ."""
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_manifest() -> pd.DataFrame:
    """Собрать по строке на изображение: имена файлов, хеши и число объектов."""
    rows = []
    # Стабильный порядок строк нужен для воспроизводимой контрольной суммы CSV.
    annotations = sorted(ANNOTATIONS_DIR.glob("*.txt"))

    if not annotations:
        raise FileNotFoundError("No annotation files found")

    for annotation_path in annotations:
        image_path = DATASET_DIR / "images" / f"{annotation_path.stem}.jpg"

        # Для каждой аннотации должен существовать парный JPG с тем же именем.
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        objects_count = 0
        vehicles_count = 0

        for line in annotation_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue

            values = list(map(int, line.split(",")))

            if len(values) != 8:
                raise ValueError(f"Invalid annotation: {annotation_path}")

            # Координаты, усечение и перекрытие не участвуют в подсчёте объектов.
            _x, _y, w, h, score, category, _truncation, _occlusion = values

            # Не учитываем игнорируемые категории и рамки с неположительным размером.
            if score == 0 or category not in VALID_CLASSES:
                continue

            if w <= 0 or h <= 0:
                continue

            objects_count += 1

            if category in VEHICLE_CLASSES:
                vehicles_count += 1

        # Хеши позволяют обнаружить изменения и изображения, и его разметки.
        rows.append(
            {
                "image": image_path.name,
                "annotation": annotation_path.name,
                "image_sha256": sha256_file(image_path),
                "annotation_sha256": sha256_file(annotation_path),
                "objects": objects_count,
                "vehicles": vehicles_count,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Сохранить CSV и связать его версию с запуском MLflow."""
    df = build_manifest()

    # Фиксируем кодировку и переводы строк, от которых зависит хеш файла.
    df.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    # Полный SHA-256 сохраняем в теге, первые 32 символа используем как digest.
    full_digest = sha256_file(MANIFEST_PATH)
    digest = full_digest[:32]

    # Регистрация датасета использует тот же сервер и эксперимент, что и EDA.
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("visdrone-eda")

    # Источник указывает на локальный манифест; его копия загружается как артефакт.
    dataset = mlflow.data.from_pandas(
        df,
        source=MANIFEST_PATH.resolve().as_uri(),
        name="visdrone2019-det-val",
        digest=digest,
    )

    # В одном run связываем входной датасет, манифест, источник и размер выборки.
    with mlflow.start_run(run_name="dataset-tracking-validation"):
        mlflow.set_tag("manifest_sha256", full_digest)
        mlflow.log_input(dataset, context="validation")

        mlflow.log_artifact(
            str(MANIFEST_PATH),
            artifact_path="dataset",
        )

        mlflow.set_tag(
            "original_source_url",
            "https://github.com/VisDrone/VisDrone-Dataset",
        )

        mlflow.log_metric("images_count", len(df))

    # Показываем идентификатор версии и сводку для быстрой проверки результата.
    print("Dataset:", dataset.name)
    print("Digest:", dataset.digest)
    print("Source:", dataset.source)
    print("Images:", len(df))
    print("Median vehicles:", df["vehicles"].median())
    print("Maximum vehicles:", df["vehicles"].max())


if __name__ == "__main__":
    main()
