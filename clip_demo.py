from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import torch

# Load the model and processor
device = "cuda"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load an image
b = 1
image = np.random.uniform(low=0, high=1, size=(5, 5, 3, 64, 64))

# Preprocess the image and convert to PyTorch tensors
inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)

# Get the image embeddings
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

print(image_features.shape) # [b, 512]