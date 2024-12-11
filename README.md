# Image Classifier

This repository contains a Python-based command-line application to train and use a deep neural network for classifying flowers. The application includes two main scripts: `train.py` for training a model and `predict.py` for predicting the class of an input image.

## Features

1. **Train** a new network on a flowers dataset and save it as a checkpoint.
2. **Predict** flower names from an image using the trained model.
3. Support for customizing model architecture, hyperparameters, and using GPU for faster training and inference.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- Pillow
- NumPy
- argparse

Install dependencies using:
```bash
pip install torch torchvision pillow numpy
```

## File Structure

- `train.py`: Script to train a new model.
- `predict.py`: Script to predict the class of an image using a trained model.
- `model_utils.py`: Contains functions for building, training, saving, and loading models.
- `image_utils.py`: Functions for preprocessing input images and making predictions.
- `data_utils.py`: Functions for loading and preprocessing datasets.

## Usage

### Training a Model
Train a new network on a dataset using the `train.py` script.

#### Basic Usage:
```bash
python train.py data_directory
```

#### Options:
- Set directory to save checkpoints:
  ```bash
  python train.py data_directory --save_dir save_directory
  ```
- Choose architecture (e.g., `vgg13`, `vgg16`, `resnet50`):
  ```bash
  python train.py data_directory --arch "vgg13"
  ```
- Set hyperparameters:
  ```bash
  python train.py data_directory --learning_rate 0.01 --hidden_units 512 --epochs 20
  ```
- Set batch size:
  ```bash
  python train.py data_directory --batch_size 64
  ```
- Use GPU for training:
  ```bash
  python train.py data_directory --gpu
  ```

### Predicting an Image
Predict the flower name and probability using the `predict.py` script.

#### Basic Usage:
```bash
python predict.py /path/to/image checkpoint
```

#### Options:
- Return top K most likely classes:
  ```bash
  python predict.py /path/to/image checkpoint --top_k 3
  ```
- Use a mapping of categories to real names:
  ```bash
  python predict.py /path/to/image checkpoint --category_names cat_to_name.json
  ```
- Use GPU for inference:
  ```bash
  python predict.py /path/to/image checkpoint --gpu
  ```

## Example Workflow

1. **Train a model**:
   ```bash
   python train.py flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 10 --batch_size 64 --gpu
   ```

2. **Predict an image**:
   ```bash
   python predict.py flowers/test/1/image_06743.jpg checkpoints/checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
   ```


