# Image Recognition Lab

Image Recognition Lab is a Streamlit app for building and testing an image
classifier. It is currently focused on logo and brand recognition, but the same
workflow can be used for any class-based image dataset.

The app lets you upload images, organize them into class folders, import raw
datasets, train a MobileNetV2 TensorFlow model, run predictions, evaluate test
accuracy, and inspect model history.

## Current Features

- Upload image files into a staging area.
- Create class labels and sort uploaded images into training, validation, and
  testing buckets.
- Scan raw image datasets and import selected classes with a train/validation/test
  split.
- Train a MobileNetV2 transfer-learning classifier.
- Save timestamped model versions plus current-model copies.
- Predict a class for a single uploaded image.
- Save corrected prediction images back into the selected training class.
- Evaluate the current model against testing images with a confusion matrix,
  classification report, and per-class accuracy table.
- View training run history and choose a previous model version as current.
- Compare model accuracy and loss across training runs with charts.
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
8. If a prediction is wrong, choose the correct class and save the image back to
   training data.
9. Use `Evaluate Model` to measure test-set performance.
10. Use `Model Manager` to compare runs and restore a previous model version.
11. Use `Dataset Health` before serious training to catch broken or suspicious
    files.

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

## Future Features

- Add a review queue for saved prediction corrections before they enter training
  data.
- Add export/import tools for sharing model bundles and dataset splits.
- Add configurable model backbones such as EfficientNet or ResNet.
- Add basic experiment notes so each training run can record dataset changes and
  manual observations.

## Change Tracking

Keep project changes in `CHANGES.txt`. Add new entries at the top with the date,
the changed files, and a short note about verification.
