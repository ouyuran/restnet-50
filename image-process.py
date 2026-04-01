from transformers import AutoImageProcessor
from PIL import Image
import numpy as np

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

image = Image.open("cat.png")
inputs = processor(image, return_tensors="np")

np.save("cat.npy", inputs["pixel_values"])