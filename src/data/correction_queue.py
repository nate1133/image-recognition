import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.utils.image_utils import clean_class_name, clean_file_name, unique_path


def queue_correction(queue_dir, class_name, image, original_filename, prediction=None):
    safe_class_name = clean_class_name(class_name)
    class_dir = Path(queue_dir) / safe_class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    image_path = unique_path(class_dir / clean_file_name(original_filename))
    image.convert("RGB").save(image_path)

    metadata = {
        "class_name": safe_class_name,
        "original_filename": original_filename,
        "prediction": prediction,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }
    image_path.with_suffix(image_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2)
    )
    return image_path


def list_corrections(queue_dir):
    items = []
    queue_dir = Path(queue_dir)
    if not queue_dir.exists():
        return items

    for path in sorted(queue_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".json"):
            continue
        metadata_path = path.with_suffix(path.suffix + ".json")
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        items.append({
            "path": path,
            "class_name": metadata.get("class_name", path.parent.name),
            "original_filename": metadata.get("original_filename", path.name),
            "prediction": metadata.get("prediction"),
            "queued_at": metadata.get("queued_at"),
        })
    return items


def review_correction(image_path, queue_dir, training_dir, approve):
    image_path = Path(image_path).resolve()
    queue_root = Path(queue_dir).resolve()
    if image_path == queue_root or queue_root not in image_path.parents:
        raise ValueError("Correction is outside the review queue.")
    if not image_path.is_file():
        raise FileNotFoundError("Queued correction no longer exists.")

    metadata_path = image_path.with_suffix(image_path.suffix + ".json")
    class_name = image_path.parent.name
    if metadata_path.exists():
        try:
            class_name = json.loads(metadata_path.read_text()).get(
                "class_name", class_name
            )
        except (json.JSONDecodeError, OSError):
            pass

    destination = None
    if approve:
        destination_dir = Path(training_dir) / clean_class_name(class_name)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = unique_path(destination_dir / image_path.name)
        shutil.move(str(image_path), destination)
    else:
        image_path.unlink()

    if metadata_path.exists():
        metadata_path.unlink()
    return destination
