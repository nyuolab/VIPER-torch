from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import torch

# Load the model and processor
device = "cuda"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load an image
b = 5
x = np.random.uniform(low=0, high=1, size=(b, 5, 3, 64, 64))
old_dim = len(x.shape)

if old_dim > 4:
    pre_shape = x.shape[:-3]
    x = x.reshape(-1, *x.shape[-3:])

# Preprocess the image and convert to PyTorch tensors
inputs = processor(images=x, return_tensors="pt", do_rescale=False).to(device)

# Get the image embeddings
with torch.no_grad():
    x = model.get_image_features(**inputs)

if old_dim > 4:
    x = x.view(*pre_shape, -1)
print(x.shape) # [b, 512]