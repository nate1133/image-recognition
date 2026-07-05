import json
from datetime import datetime
import pandas as pd


def train_image_classifier(
    training_dir,
    validation_dir,
    model_dir,
    image_size=160,
    batch_size=32,
    epochs=5,
):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model_dir.mkdir(parents=True, exist_ok=True)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"logo_classifier_{run_time}.keras"

    model_path = model_dir / model_filename
    latest_model_path = model_dir / "logo_classifier.keras"

    classes_path = model_dir / f"classes_{run_time}.json"
    latest_classes_path = model_dir / "classes.json"

    info_path = model_dir / f"model_info_{run_time}.json"
    latest_info_path = model_dir / "model_info.json"

    training_log_path = model_dir / "training_runs.csv"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        training_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        class_names=class_names,
    )

    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

    base_model = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    model.save(model_path)
    model.save(latest_model_path)

    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=4)

    with open(latest_classes_path, "w") as f:
        json.dump(class_names, f, indent=4)

    model_info = {
        "run_time": run_time,
        "model_name": model_filename,
        "architecture": "MobileNetV2",
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "classes": class_names,
        "num_classes": num_classes,
        "final_training_accuracy": float(history.history["accuracy"][-1]),
        "final_validation_accuracy": float(history.history["val_accuracy"][-1]),
        "final_training_loss": float(history.history["loss"][-1]),
        "final_validation_loss": float(history.history["val_loss"][-1]),
    }

    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=4)

    with open(latest_info_path, "w") as f:
        json.dump(model_info, f, indent=4)

    log_row = pd.DataFrame([{
        "run_time": run_time,
        "model_file": model_filename,
        "architecture": "MobileNetV2",
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_classes": num_classes,
        "training_accuracy": model_info["final_training_accuracy"],
        "validation_accuracy": model_info["final_validation_accuracy"],
        "training_loss": model_info["final_training_loss"],
        "validation_loss": model_info["final_validation_loss"],
    }])

    if training_log_path.exists():
        old_log = pd.read_csv(training_log_path)
        updated_log = pd.concat([old_log, log_row], ignore_index=True)
    else:
        updated_log = log_row

    updated_log.to_csv(training_log_path, index=False)

    history_df = pd.DataFrame({
        "Epoch": list(range(1, epochs + 1)),
        "Training Accuracy": history.history["accuracy"],
        "Validation Accuracy": history.history["val_accuracy"],
        "Training Loss": history.history["loss"],
        "Validation Loss": history.history["val_loss"],
    })

    return model_info, history_df
