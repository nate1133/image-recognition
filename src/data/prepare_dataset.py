from pathlib import Path
import random
import shutil
from collections import defaultdict

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def get_direct_images(folder: Path):
    return [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def clear_folder(folder: Path):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def import_selected_classes(
    selected_rows,
    training_dir,
    validation_dir,
    testing_dir,
    train_pct=0.8,
    val_pct=0.1,
    test_pct=0.1,
    seed=42,
    clear_existing=False
):
    random.seed(seed)

    training_dir = Path(training_dir)
    validation_dir = Path(validation_dir)
    testing_dir = Path(testing_dir)

    grouped = defaultdict(list)

    for row in selected_rows:
        class_name = row["class"]
        folder_path = Path(row["path"])

        images = get_direct_images(folder_path)
        grouped[class_name].extend(images)

    summary = []

    for class_name, images in grouped.items():
        random.shuffle(images)

        total = len(images)

        n_train = int(total * train_pct)
        n_val = int(total * val_pct)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        train_dest = training_dir / class_name
        val_dest = validation_dir / class_name
        test_dest = testing_dir / class_name

        if clear_existing:
            clear_folder(train_dest)
            clear_folder(val_dest)
            clear_folder(test_dest)
        else:
            train_dest.mkdir(parents=True, exist_ok=True)
            val_dest.mkdir(parents=True, exist_ok=True)
            test_dest.mkdir(parents=True, exist_ok=True)

        def copy_images(image_list, dest_folder):
            copied = 0

            for img in image_list:
                destination = dest_folder / img.name
                counter = 1

                if destination.exists():
                    while destination.exists():
                        destination = dest_folder / f"{img.stem}_{counter}{img.suffix}"
                        counter += 1

                shutil.copy2(img, destination)
                copied += 1

            return copied

        train_count = copy_images(train_imgs, train_dest)
        val_count = copy_images(val_imgs, val_dest)
        test_count = copy_images(test_imgs, test_dest)

        summary.append({
            "class": class_name,
            "total": total,
            "training": train_count,
            "validation": val_count,
            "testing": test_count,
        })

    return summary
