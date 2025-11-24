# src/train_resnet_features.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from utils import load_cat_to_name, save_class_names
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "flower_data", "train")
CAT_TO_NAME = os.path.join(PROJECT_ROOT, "data", "flower_data", "cat_to_name.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)

print("Loading ResNet50 backbone...")
base = ResNet50(include_top=False, pooling="avg", input_shape=IMG_SIZE+(3,))
base.trainable = False


def extract_resnet_feature(img_path):
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = base.predict(x, verbose=0)
        return feat[0]
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None


X = []
y = []
folder_names = sorted(os.listdir(DATA_DIR))
cat_to_name = load_cat_to_name(CAT_TO_NAME)
human_names = []

print("Extracting ResNet features...")

for label, folder in enumerate(folder_names):
    human_names.append(cat_to_name.get(folder, folder))

    folder_path = os.path.join(DATA_DIR, folder)
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for fname in tqdm(image_files, desc=f"Folder {folder}", ncols=80):
        fpath = os.path.join(folder_path, fname)
        feat = extract_resnet_feature(fpath)
        if feat is not None:
            X.append(feat)
            y.append(label)

X = np.array(X)
y = np.array(y)
print("Feature shape:", X.shape)

save_class_names(human_names, os.path.join(PROJECT_ROOT, "class_names_resnet.json"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = models.Sequential([
    layers.Input(shape=(2048,)),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(human_names), activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training FCN...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32
)

model.save(os.path.join(OUT_DIR, "resnet_features_model.h5"))
print("Model saved: resnet_features_model.h5")
