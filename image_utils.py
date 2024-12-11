from PIL import Image
import numpy as np
import torch
import os

def process_image(image_path, gpu=False):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a tensor moved to the specified device."""

    # Type validation for image path
    if not isinstance(image_path, str):
        raise TypeError("image_path must be a string.")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Open the image
    image = Image.open(image_path).convert('RGB')

    # Resize the image to maintain aspect ratio with the shortest side as 256 pixels
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))

    # Crop the center of the image to a 224x224 square
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Convert image to a numpy array and normalize the values
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions for PyTorch (C x H x W)
    np_image = np_image.transpose((2, 0, 1))

    # Convert to tensor
    tensor_image = torch.tensor(np_image, dtype=torch.float32)

    # Validate output dimensions
    if tensor_image.shape != (3, 224, 224):
        raise ValueError("Processed image must have dimensions (3, 224, 224).")

    # Move to GPU if specified
    if gpu and torch.cuda.is_available():
        tensor_image = tensor_image.to("cuda")

    return tensor_image