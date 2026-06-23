from pathlib import Path
from datetime import datetime
import pandas as pd
from PIL import Image
import imagehash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = PROJECT_ROOT / "data" / "labels" / "image_metadata.csv"


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_image_metadata(image_path: Path) -> dict:
    """
    Reads one image and returns useful metadata.
    """
    label = image_path.parent.name

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            file_hash = str(imagehash.average_hash(img))

        return {
            "file_name": image_path.name,
            "file_path": str(image_path.relative_to(PROJECT_ROOT)),
            "label": label,
            "file_extension": image_path.suffix.lower(),
            "width": width,
            "height": height,
            "image_hash": file_hash,
            "file_size_bytes": image_path.stat().st_size,
            "scan_status": "success",
            "error_message": "",
            "scanned_at": datetime.now().isoformat(timespec="seconds"),
        }

    except Exception as e:
        return {
            "file_name": image_path.name,
            "file_path": str(image_path.relative_to(PROJECT_ROOT)),
            "label": label,
            "file_extension": image_path.suffix.lower(),
            "width": None,
            "height": None,
            "image_hash": None,
            "file_size_bytes": image_path.stat().st_size,
            "scan_status": "failed",
            "error_message": str(e),
            "scanned_at": datetime.now().isoformat(timespec="seconds"),
        }


def scan_images() -> pd.DataFrame:
    """
    Scans all category folders inside data/raw.
    """
    records = []

    for image_path in RAW_IMAGE_DIR.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
            records.append(get_image_metadata(image_path))

    return pd.DataFrame(records)


def main():
    df = scan_images()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Scanned {len(df)} image files.")
    print(f"Metadata saved to: {OUTPUT_FILE}")

    if not df.empty:
        print("\nImages by label:")
        print(df["label"].value_counts())

        duplicate_count = df.duplicated(subset=["image_hash"]).sum()
        print(f"\nPossible duplicate images: {duplicate_count}")


if __name__ == "__main__":
    main()
