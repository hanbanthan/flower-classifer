# Flower Classifier

An image classifier for recognizing **102 flower categories** using TensorFlow and transfer learning. The project uses a frozen ImageNet-pretrained MobileNetV2 feature extractor and a small trainable classifier.

## Features

- Loads the Oxford Flowers 102 dataset from local `train`, `valid`, and `test` folders.
- Resizes and normalizes images to `224 x 224` pixels.
- Trains a 102-class classifier with early stopping.
- Saves the trained model as `flower_classifier_model.keras`.
- Predicts the top `k` flower classes for a new image.

## Project structure

```text
.
├── flower-classifier.ipynb       # Dataset exploration and model training
├── predict.py                    # Command-line inference script
├── flower_classifier_model.keras # Trained model
├── requirements.txt              # Python dependencies
├── flower_data/
│   ├── cat_to_name.json          # Class ID to flower name mapping
│   ├── train/
│   ├── valid/
│   └── test/
└── test_images/                  # Example images for inference
```

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install tensorflow pillow numpy matplotlib
```

## Run a prediction

From the project directory, provide the image path and model path:

```bash
python predict.py test_images/pink-primrose.jpg flower_classifier_model.keras
```

The script prints the five most likely classes by default and displays the input image with the top prediction.

### Options

```bash
python predict.py <image_path> <model_path> [options]
```

| Option | Default | Description |
| --- | --- | --- |
| `--top_k` | `5` | Number of predictions to return |
| `--category_names` | `flower_data/cat_to_name.json` | Path to the class-name mapping |

Example:

```bash
python predict.py test_images/hard-leaved_pocket_orchid.jpg flower_classifier_model.keras --top_k 3
```

Example output:

```text
Top 3 Predictions:
hard-leaved pocket orchid: 0.9876
...
```

## Train the model

Open [flower-classifier.ipynb](flower-classifier.ipynb) in Jupyter Notebook or VS Code and run the cells in order. The notebook:

1. Loads the local dataset.
2. Prepares the training, validation, and test datasets.
3. Builds a MobileNetV2-based classifier with 102 output classes.
4. Trains for up to 30 epochs with early stopping.
5. Evaluates the model and saves it to `flower_classifier_model.keras`.

The saved model can then be used with `predict.py`.

## Model details

- Base network: MobileNetV2 with ImageNet weights
- Input shape: `(224, 224, 3)`
- Trainable layers: global average pooling, dense layer with 1024 units, dropout, and 102-class softmax output
- Optimizer: Adam with learning rate `0.0005`
- Loss: sparse categorical cross-entropy
- Training result recorded in the notebook: validation accuracy reached approximately `90%`

## Dataset

This project uses the [Oxford Flowers 102 dataset](https://www.kaggle.com/datasets/waseemalastal/the-oxford-flowers-102-dataset). The class names are stored in `flower_data/cat_to_name.json`.

## Notes

- Run commands from the repository root so the default relative paths resolve correctly.
- The model expects RGB images. `predict.py` converts input images to RGB automatically.
- Prediction preprocessing must match training preprocessing: resize to `224 x 224` and scale pixel values to `[0, 1]`.