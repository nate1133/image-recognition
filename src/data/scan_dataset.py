from pathlib import Path

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def scan_dataset(raw_dir):
    raw_dir = Path(raw_dir)

    results = []

    for folder in raw_dir.rglob("*"):
        if not folder.is_dir():
            continue

        image_count = count_images(folder)

        if image_count == 0:
            continue

        results.append({
            "dataset": folder.relative_to(raw_dir).parts[0],
            "class": folder.name,
            "path": str(folder),
            "images": image_count,
        })

    results.sort(key=lambda x: x["images"], reverse=True)

    return results
