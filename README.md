# Image Recognition Lab

Image Recognition Lab is a Streamlit app for building and testing an image
classifier. It is currently focused on logo and brand recognition, but the same
workflow can be used for any class-based image dataset.

The app lets you upload images, organize them into class folders, import raw
datasets, train TensorFlow transfer-learning models with several backbones, run
predictions, review corrections, evaluate test accuracy, and share model and
dataset bundles.

## Current Features

- Upload image files into a staging area.
- Create class labels and sort uploaded images into training, validation, and
  testing buckets.
- Scan raw image datasets and import selected classes with a train/validation/test
  split.
- Train a MobileNetV2, EfficientNetB0, or ResNet50 transfer-learning classifier.
- Save timestamped model versions plus current-model copies.
- Predict a class for a single uploaded image.
- Queue corrected predictions for approval or rejection before training use.
- Export and import portable ZIP bundles containing the current model and/or
  training, validation, and testing splits.
- Evaluate the current model against testing images with a confusion matrix,
  classification report, and per-class accuracy table.
- View training run history and choose a previous model version as current.
- Compare model accuracy and loss across training runs with charts.
- Record dataset changes and manual observations with each training run, then
  review those notes in Model Manager.
- Review per-class training sample counts and imbalance warnings before
  training.
- Run dataset health checks for broken images, tiny images, wrong file types,
  empty class folders, duplicate filenames, and perceptual duplicate images.
- Run automated tests for dataset import, prediction correction, model metadata
  loading, and duplicate-image detection.

## Project Layout

```text
image-recognition-lab/
  app/
    main.py                    Streamlit entrypoint
    pages/                     Streamlit tab implementations
    utils/                     Shared paths and image helpers
  data/
    raw/                       Raw source datasets
    uploads/                   Uploaded images waiting to be sorted
    training/                  Training images grouped by class
    validation/                Validation images grouped by class
    testing/                   Testing images grouped by class
    correction_review/         Prediction corrections awaiting review
  models/                      Current trained model files used by the app
  models_output/               Older or exported model artifacts
  src/
    data/                      Dataset scanning/import helpers
    models/                    Training, prediction, and evaluation logic
  CHANGES.txt                  Manual change log
  requirements.txt             Python dependencies
```

## Setup

From the project folder:

```bash
cd /home/homeserver/projects/image-recognition-lab
```

If the existing virtual environment is already available, use it:

```bash
source venv/bin/activate
```

If you need to create a fresh environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run app/main.py --server.port 8502
```

Then open the local Streamlit URL shown in the terminal.

## Typical Workflow

1. Open the app and go to `Upload Images`.
2. Upload image files you want to label.
3. Go to `Dataset Manager`.
4. Create class folders, then move uploaded images into training, validation, or
   testing.
5. Use `Raw Dataset Scanner` when importing a larger folder or ZIP dataset.
6. Go to `Train Model` and train a new classifier.
7. Use `Predict` to test individual images.
8. If a prediction is wrong, submit the correction, then approve it in
   `Correction Review`.
9. Use `Evaluate Model` to measure test-set performance.
10. Use `Model Manager` to compare runs and restore a previous model version.
11. Use `Dataset Health` before serious training to catch broken or suspicious
    files.
12. Use `Export / Import` to share or restore a model and dataset splits.

## Configurable Model Backbones

The `Train Model` tab offers three pretrained backbones:

- `MobileNetV2`: the default and lightest option, suitable for quick experiments
  and lower-powered systems.
- `EfficientNetB0`: a good balance between model size and classification
  performance.
- `ResNet50`: a larger, well-established architecture that generally requires
  more training time and memory.

Choose the backbone along with image size, batch size, and epoch count before
starting a run. The selected architecture is saved in the run metadata and
training history. Prediction and evaluation automatically use the preprocessing
required by that architecture.

The pretrained backbone is frozen during training. Only the classification head
is trained against the classes in the current dataset.

## Prediction Correction Review

Predicted images no longer enter the training dataset immediately:

1. Run a prediction in the `Predict` tab.
2. Select the correct class if the prediction is wrong.
3. Click `Submit Correction for Review`.
4. Open `Correction Review` to inspect the image, original prediction, and
   proposed class.
5. Choose `Approve` to move it into the matching training class, or `Reject` to
   discard it.

Pending images and their metadata are stored under
`data/correction_review/<class>/`. Approved images are moved to
`data/training/<class>/` with unique filenames, so an existing training image is
not overwritten.

## Export and Import

The `Export / Import` tab creates portable ZIP bundles for sharing or backup.
An export can include:

- The current model, class list, and model metadata.
- The training, validation, and testing dataset splits.
- Both the model and dataset in one bundle.

During import, choose whether to restore the model, the dataset, or both.
Existing files are preserved by default; enable `Overwrite files with matching
names` only when the imported copies should replace them. Bundle paths are
validated during import to prevent files from being written outside the model
and dataset directories.

Only current-model files are included in the model portion of a bundle. Older
timestamped model versions and the complete training-run history remain local.

## Experiment Notes

Each run can include two optional notes in the `Train Model` tab:

- `Dataset changes`: additions, removals, relabeling, cleanup, or split changes
  made since the previous experiment.
- `Manual observations`: hypotheses, known limitations, visual patterns,
  confusion between classes, or follow-up ideas.

Notes are saved in both the timestamped `model_info` JSON and
`models/training_runs.csv`. Open `Model Manager` and expand a run under
`Experiment Notes` to review them alongside model metrics.

## Model Files

Training writes timestamped model artifacts and updates the current model files:

- `models/logo_classifier.keras`
- `models/classes.json`
- `models/model_info.json`
- `models/training_runs.csv`

The prediction and evaluation pages use the current files in `models/`.

## Development Checks

Compile-check the app and support modules:

```bash
./venv/bin/python -m compileall app src
```

Run the automated tests:

```bash
./venv/bin/python -m unittest discover -v
```

Run a quick metadata load check:

```bash
./venv/bin/python -c "from app.utils.paths import MODEL_DIR; from src.models.predict_model import load_model_metadata; print(load_model_metadata(MODEL_DIR)[:3])"
```

## Troubleshooting

- `No trained model found`: train a model first, or confirm that
  `models/logo_classifier.keras` exists.
- `classes.json is missing`: train again or restore the matching class metadata
  from a timestamped model run.
- Training fails with class mismatch: make sure validation folders match the
  training class folders.
- Evaluation fails with class mismatch: make sure testing folders match the
  model classes in `models/classes.json`.
- TensorFlow says CUDA is missing: the app can still run on CPU; GPU support is
  optional.
- `cannot import name 'CORRECTION_REVIEW_DIR'`: fully stop and restart Streamlit
  so it reloads `app/utils/paths.py`; refreshing the browser does not clear
  cached Python modules.

## Change Tracking

Keep project changes in `CHANGES.txt`. Add new entries at the top with the date,
the changed files, and a short note about verification.
