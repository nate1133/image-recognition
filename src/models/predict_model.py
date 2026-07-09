import json
import numpy as np
from PIL import Image

from app.utils.image_utils import clean_file_name, unique_path


def load_model_metadata(model_dir):
    model_path = model_dir / "logo_classifier.keras"
    classes_path = model_dir / "classes.json"
    info_path = model_dir / "model_info.json"

    if not model_path.exists():
        raise FileNotFoundError("No trained model found.")

    if not classes_path.exists():
        raise FileNotFoundError("classes.json is missing.")

    with open(classes_path, "r") as f:
        class_names = json.load(f)

    image_size = 160
    model_info = {}

    if info_path.exists():
        with open(info_path, "r") as f:
            model_info = json.load(f)

        image_size = model_info.get("image_size", 160)

    return model_path, class_names, image_size, model_info


def predict_image(model_dir, image_file):
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    model_path, class_names, image_size, model_info = load_model_metadata(model_dir)

    model = tf.keras.models.load_model(model_path)

    img = Image.open(image_file).convert("RGB")
    img_resized = img.resize((image_size, image_size))

    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]

    top_index = int(np.argmax(predictions))
    top_class = class_names[top_index]
    top_confidence = float(predictions[top_index])

    results = []

    top_n = min(5, len(class_names))
    top_indices = predictions.argsort()[-top_n:][::-1]

    for idx in top_indices:
        results.append({
            "Class": class_names[int(idx)],
            "Confidence": float(predictions[int(idx)]),
            "Confidence %": float(predictions[int(idx)]) * 100,
        })

    return {
        "image": img,
        "top_class": top_class,
        "top_confidence": top_confidence,
        "results": results,
        "class_names": class_names,
        "image_size": image_size,
        "model_info": model_info,
    }


def save_prediction_correction(training_dir, class_name, image, original_filename):
    correction_dir = training_dir / class_name
    correction_dir.mkdir(parents=True, exist_ok=True)

    save_path = unique_path(correction_dir / clean_file_name(original_filename))
    image.convert("RGB").save(save_path)

    return save_path
