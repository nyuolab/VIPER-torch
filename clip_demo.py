from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load an image
image = Image.open("path/to/your/image.jpg")

# Preprocess the image and convert to PyTorch tensors
inputs = processor(images=image, return_tensors="pt")

# Get the image embeddings
with torch.no_grad():
    image_features = model.get_image_features(**inputs)