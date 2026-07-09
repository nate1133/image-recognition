import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


SPLITS = ("training", "validation", "testing")
MODEL_FILES = ("logo_classifier.keras", "classes.json", "model_info.json")


def _write_tree(archive, root, prefix):
    root = Path(root)
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            archive.write(path, f"{prefix}/{path.relative_to(root).as_posix()}")


def create_portable_bundle(model_dir, split_dirs, include_model=True, include_dataset=True):
    output = io.BytesIO()
    manifest = {
        "format": "image-recognition-lab-bundle",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "includes_model": include_model,
        "includes_dataset": include_dataset,
    }
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        if include_model:
            for name in MODEL_FILES:
                path = Path(model_dir) / name
                if path.exists():
                    archive.write(path, f"model/{name}")
        if include_dataset:
            for split in SPLITS:
                if split in split_dirs:
                    _write_tree(archive, split_dirs[split], f"dataset/{split}")
    return output.getvalue()


def _safe_members(archive):
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in bundle: {info.filename}")
        yield info, path


def import_portable_bundle(bundle, model_dir, split_dirs, import_model=True,
                           import_dataset=True, overwrite=False):
    data = bundle.read() if hasattr(bundle, "read") else bundle
    imported = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError("Not a valid Image Recognition Lab bundle.") from exc
        if manifest.get("format") != "image-recognition-lab-bundle":
            raise ValueError("Unsupported bundle format.")

        for info, path in _safe_members(archive):
            if info.is_dir() or path.name == "manifest.json":
                continue
            destination = None
            if import_model and len(path.parts) == 2 and path.parts[0] == "model":
                if path.name in MODEL_FILES:
                    destination = Path(model_dir) / path.name
            elif import_dataset and len(path.parts) >= 3 and path.parts[0] == "dataset":
                split = path.parts[1]
                if split in split_dirs:
                    destination = Path(split_dirs[split]).joinpath(*path.parts[2:])
            if destination is None:
                continue
            if destination.exists() and not overwrite:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            imported.append(destination)
    return imported
