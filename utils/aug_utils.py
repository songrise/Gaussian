import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

def apply_perspective(img, matrix):
    """
    Apply a perspective transformation to an image tensor.

    Parameters:
    img (Tensor): image tensor of shape (C, H, W)
    matrix (Tensor): a perspective transformation matrix of shape (3, 4)

    Returns:
    Tensor: Transformed image.
    """
    # Convert PIL Image to Tensor if necessary
    if isinstance(img, Image.Image):
        transform_to_tensor = T.ToTensor()
        img = transform_to_tensor(img)

    # Get the image height and width
    _, h, w = img.size()


    # Create normalized grid
    grid = F.affine_grid(matrix.unsqueeze(0), img.unsqueeze(0).size(), align_corners=False)

    # Sample the grid
    output = F.grid_sample(img.unsqueeze(0), grid, align_corners=False)
    return output.squeeze()

# Example usage:
# Define your 3x4 perspective transformation matrix
# Note that the last row should be [0, 0, 1] for affine transformations
# Define the scaling factors
# Convert angle from degrees to radians
import math
angle = 1  # degrees
angle_rad = math.radians(angle)  # radians

# Create the 2x3 affine transformation matrix for rotation
cos_a = math.cos(angle_rad)
sin_a = math.sin(angle_rad)

affine_matrix = torch.tensor([[cos_a, -sin_a, 0],
                              [sin_a, cos_a, 0]])

# Open an image and convert it to a tensor
img = Image.open("/root/autodl-tmp/gaussian-splatting/data/truck/images/000001.jpg")
img_tensor = T.ToTensor()(img)

# Apply the transformation
transformed_tensor = apply_perspective(img_tensor, affine_matrix)

# Convert the tensor to PIL Image and save or display
transformed_image = T.ToPILImage()(transformed_tensor)
transformed_image.save("test.jpg")