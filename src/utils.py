# src/utils.py
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

def load_cat_to_name(path="../data/cat_to_name.json"):
    """Load mapping folder-id -> human-name"""
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # keys in file are strings "1","2",..., ensure kept as strings
    return mapping

def save_class_names(class_names, path="../class_names.json"):
    """Save list of class names (human readable)"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

def load_class_names(path="../class_names.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_image_for_model(img_path, target_size=(224,224)):
    """Load image, return a batch (1, H, W, 3) ready for model.predict."""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)   # EfficientNet preprocessing
    return x
