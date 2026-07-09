import json
import pandas as pd
import numpy as np


def get_class_dirs(folder):
    return sorted(path.name for path in folder.iterdir() if path.is_dir())


def evaluate_image_classifier(model_dir, testing_dir):
    model_path = model_dir / "logo_classifier.keras"
    classes_path = model_dir / "classes.json"
    info_path = model_dir / "model_info.json"

    if not model_path.exists():
        raise FileNotFoundError("No trained model found.")

    if not classes_path.exists():
        raise FileNotFoundError("classes.json is missing.")

    with open(classes_path, "r") as f:
        class_names = json.load(f)

    testing_class_names = get_class_dirs(testing_dir)

    if set(testing_class_names) != set(class_names):
        missing = sorted(set(class_names) - set(testing_class_names))
        extra = sorted(set(testing_class_names) - set(class_names))

        details = []

        if missing:
            details.append(f"missing test folders: {', '.join(missing)}")

        if extra:
            details.append(f"extra test folders: {', '.join(extra)}")

        raise ValueError(
            "Testing folder classes must match model classes before evaluation"
            + (f" ({'; '.join(details)})" if details else ".")
        )

    image_size = 160
    architecture = "MobileNetV2"

    if info_path.exists():
        with open(info_path, "r") as f:
            model_info = json.load(f)
            image_size = model_info.get("image_size", 160)
            architecture = model_info.get("architecture", "MobileNetV2")

    import tensorflow as tf
    from src.models.backbones import get_preprocess_input
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    preprocess_input = get_preprocess_input(architecture)

    model = tf.keras.models.load_model(model_path)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        testing_dir,
        image_size=(image_size, image_size),
        batch_size=32,
        shuffle=False,
        label_mode="int",
        class_names=class_names,
    )

    test_class_names = test_ds.class_names

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        processed_images = preprocess_input(images)
        predictions = model.predict(processed_images, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    accuracy = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {name}" for name in test_class_names],
        columns=[f"Predicted: {name}" for name in test_class_names],
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=test_class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    class_accuracy_rows = []

    for i, class_name in enumerate(test_class_names):
        correct = cm[i, i]
        total = cm[i].sum()
        class_accuracy = correct / total if total > 0 else 0

        class_accuracy_rows.append({
            "Class": class_name,
            "Correct": int(correct),
            "Total": int(total),
            "Accuracy %": class_accuracy * 100,
        })

    class_accuracy_df = pd.DataFrame(class_accuracy_rows)

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm_df,
        "classification_report": report_df,
        "class_accuracy": class_accuracy_df,
        "model_classes": class_names,
        "testing_classes": test_class_names,
        "image_size": image_size,
    }
