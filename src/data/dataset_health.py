from pathlib import Path

import imagehash
from PIL import Image


def scan_dataset_health(scan_dirs, image_exts, tiny_threshold=100):
    broken_images = []
    tiny_images = []
    wrong_file_types = []
    empty_class_folders = []
    seen_names = {}
    seen_hashes = {}

    for bucket_name, bucket_dir in scan_dirs.items():
        bucket_dir = Path(bucket_dir)

        if not bucket_dir.exists():
            continue

        for path in bucket_dir.rglob("*"):
            if path.is_dir():
                if path != bucket_dir and not any(path.iterdir()):
                    empty_class_folders.append({
                        "Bucket": bucket_name,
                        "Folder": str(path),
                    })
                continue

            if not path.is_file():
                continue

            if path.suffix.lower() not in image_exts:
                wrong_file_types.append({
                    "Bucket": bucket_name,
                    "File": str(path),
                    "Extension": path.suffix,
                })
                continue

            seen_names.setdefault(path.name, []).append(str(path))

            try:
                with Image.open(path) as img:
                    width, height = img.size

                    if width < tiny_threshold or height < tiny_threshold:
                        tiny_images.append({
                            "Bucket": bucket_name,
                            "File": str(path),
                            "Width": width,
                            "Height": height,
                        })

                    image_hash = str(imagehash.average_hash(img))
                    seen_hashes.setdefault(image_hash, []).append({
                        "Bucket": bucket_name,
                        "File": str(path),
                    })

                    img.verify()

            except Exception:
                broken_images.append({
                    "Bucket": bucket_name,
                    "File": str(path),
                })

    duplicate_filenames = {
        name: locations
        for name, locations in seen_names.items()
        if len(locations) > 1
    }

    duplicate_images = {
        image_hash: locations
        for image_hash, locations in seen_hashes.items()
        if len(locations) > 1
    }

    return {
        "broken_images": broken_images,
        "tiny_images": tiny_images,
        "wrong_file_types": wrong_file_types,
        "empty_class_folders": empty_class_folders,
        "duplicate_filenames": duplicate_filenames,
        "duplicate_images": duplicate_images,
    }
