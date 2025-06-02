import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import random


def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 128, 128, 1))
    return img_array

def predict_random_image(model, test_dir):
    # Test predictions with the base dataset
    model = load_model(model)

    # Get all class folders
    class_folders = [os.path.join(test_dir, cls) for cls in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, cls))]

    # Choose a random class folder
    random_class = random.choice(class_folders)

    # Choose a random image file from that class
    image_file = random.choice(os.listdir(random_class))

    # Full path to the image
    image_path = os.path.join(random_class, image_file)

    print("Selected image:", image_path)

    # Preprocess the image
    image = preprocess_image(image_path)

    # Make a prediction
    prediction = model.predict(image)
    predicted_class = prediction.argmax(axis=1)[0]

    print(f"Predicted class index: {predicted_class}")