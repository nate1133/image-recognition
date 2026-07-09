import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from app.pages.model_manager import prepare_model_comparison_df
from app.pages.train import get_training_balance_warnings
from src.data.dataset_health import scan_dataset_health
from src.data.prepare_dataset import import_selected_classes
from src.models.predict_model import (
    load_model_metadata,
    save_prediction_correction,
)


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def make_image(path: Path, color=(255, 0, 0), size=(120, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def make_split_image(path: Path, left_color, right_color, size=(120, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, left_color)

    for x in range(size[0] // 2, size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), right_color)

    image.save(path)


class DatasetImportTests(unittest.TestCase):
    def test_import_selected_classes_splits_and_copies_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_class = root / "raw" / "logos" / "Nike"

            for index in range(5):
                make_image(raw_class / f"logo_{index}.jpg", color=(index, 0, 0))

            summary = import_selected_classes(
                selected_rows=[{
                    "class": "Nike",
                    "path": str(raw_class),
                }],
                training_dir=root / "training",
                validation_dir=root / "validation",
                testing_dir=root / "testing",
                train_pct=0.6,
                val_pct=0.2,
                test_pct=0.2,
                seed=1,
            )

            self.assertEqual(summary, [{
                "class": "Nike",
                "total": 5,
                "training": 3,
                "validation": 1,
                "testing": 1,
            }])
            self.assertEqual(len(list((root / "training" / "Nike").glob("*.jpg"))), 3)
            self.assertEqual(len(list((root / "validation" / "Nike").glob("*.jpg"))), 1)
            self.assertEqual(len(list((root / "testing" / "Nike").glob("*.jpg"))), 1)


class PredictionCorrectionTests(unittest.TestCase):
    def test_save_prediction_correction_creates_unique_training_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            training_dir = Path(tmpdir) / "training"
            image = Image.new("RGB", (120, 120), (0, 255, 0))

            first_path = save_prediction_correction(
                training_dir=training_dir,
                class_name="Toyota",
                image=image,
                original_filename="../test logo.png",
            )
            second_path = save_prediction_correction(
                training_dir=training_dir,
                class_name="Toyota",
                image=image,
                original_filename="../test logo.png",
            )

            self.assertTrue(first_path.exists())
            self.assertTrue(second_path.exists())
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(first_path.name, "test logo.png")
            self.assertEqual(second_path.name, "test logo_1.png")


class ModelMetadataTests(unittest.TestCase):
    def test_load_model_metadata_reads_classes_and_image_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "logo_classifier.keras").write_text("placeholder")
            (model_dir / "classes.json").write_text(json.dumps(["Audi", "BMW"]))
            (model_dir / "model_info.json").write_text(json.dumps({
                "image_size": 224,
                "architecture": "MobileNetV2",
            }))

            model_path, class_names, image_size, model_info = load_model_metadata(model_dir)

            self.assertEqual(model_path, model_dir / "logo_classifier.keras")
            self.assertEqual(class_names, ["Audi", "BMW"])
            self.assertEqual(image_size, 224)
            self.assertEqual(model_info["architecture"], "MobileNetV2")


class DatasetHealthTests(unittest.TestCase):
    def test_scan_dataset_health_detects_perceptual_duplicate_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            training = root / "training"
            testing = root / "testing"

            make_split_image(
                training / "Nike" / "logo_a.jpg",
                left_color=(25, 50, 75),
                right_color=(200, 210, 220),
            )
            make_split_image(
                testing / "Nike" / "different_name.jpg",
                left_color=(25, 50, 75),
                right_color=(200, 210, 220),
            )
            make_split_image(
                testing / "Nike" / "unique.jpg",
                left_color=(200, 210, 220),
                right_color=(25, 50, 75),
            )

            results = scan_dataset_health(
                scan_dirs={
                    "Training": training,
                    "Testing": testing,
                },
                image_exts=IMAGE_EXTS,
            )

            self.assertEqual(results["broken_images"], [])
            self.assertEqual(results["wrong_file_types"], [])
            self.assertEqual(results["duplicate_filenames"], {})
            self.assertEqual(len(results["duplicate_images"]), 1)

            duplicate_locations = next(iter(results["duplicate_images"].values()))
            duplicate_files = {Path(location["File"]).name for location in duplicate_locations}

            self.assertEqual(duplicate_files, {"logo_a.jpg", "different_name.jpg"})


class TrainingBalanceTests(unittest.TestCase):
    def test_get_training_balance_warnings_detects_low_and_imbalanced_classes(self):
        dataset_df = pd.DataFrame([
            {"Class": "large", "Training Images": 90},
            {"Class": "small", "Training Images": 9},
            {"Class": "empty", "Training Images": 0},
        ])

        warnings = get_training_balance_warnings(dataset_df)

        self.assertEqual(len(warnings), 3)
        self.assertIn("empty", warnings[0])
        self.assertIn("small (9)", warnings[1])
        self.assertIn("largest class has 90 images", warnings[2])


class ModelComparisonTests(unittest.TestCase):
    def test_prepare_model_comparison_df_orders_runs_and_numeric_columns(self):
        runs_df = pd.DataFrame([
            {
                "run_time": "20260707_2",
                "model_file": "second.keras",
                "training_accuracy": "0.8",
                "validation_accuracy": "0.7",
                "training_loss": "0.5",
                "validation_loss": "0.6",
            },
            {
                "run_time": "20260707_1",
                "model_file": "first.keras",
                "training_accuracy": "0.9",
                "validation_accuracy": "0.75",
                "training_loss": "0.4",
                "validation_loss": "0.55",
            },
        ])

        comparison_df = prepare_model_comparison_df(runs_df)

        self.assertEqual(
            list(comparison_df.index),
            [
                "20260707_1 - first.keras",
                "20260707_2 - second.keras",
            ],
        )
        self.assertEqual(
            comparison_df.loc["20260707_1 - first.keras", "training_accuracy"],
            0.9,
        )
        self.assertEqual(
            comparison_df.loc["20260707_2 - second.keras", "validation_loss"],
            0.6,
        )


if __name__ == "__main__":
    unittest.main()
