# src/train_hog_svm.py
import os
import joblib
import numpy as np
from skimage import io, color
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import load_cat_to_name, save_class_names
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "flower_data", "train")
CAT_TO_NAME = os.path.join(PROJECT_ROOT, "data", "flower_data", "cat_to_name.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading class mapping...")
cat_to_name = load_cat_to_name(CAT_TO_NAME)

# Ảnh resize về 128x128 (nhanh hơn nhiều so với kích thước gốc)
IMG_SIZE = (128, 128)

def extract_hog(img_path):
    # Đọc ảnh
    img = io.imread(img_path)

    # Nếu ảnh grayscale, tự động chuyển về RGB
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # Resize đồng nhất
    img = resize(img, IMG_SIZE, anti_aliasing=True)

    # Chuyển sang grayscale
    img = color.rgb2gray(img)

    # Tính HOG
    feature = hog(
        img,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return feature

X = []
y = []
folder_names = sorted(os.listdir(DATA_DIR))

print("Extracting HOG features...")

for label, folder in enumerate(folder_names):
    folder_path = os.path.join(DATA_DIR, folder)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Thêm progress bar
    for fname in tqdm(image_files, desc=f"Folder {folder}", ncols=80):
        fpath = os.path.join(folder_path, fname)
        feat = extract_hog(fpath)
        X.append(feat)
        y.append(label)

# Convert thành numpy array chắc chắn 2D
X = np.array(X, dtype="float32")
y = np.array(y)

print("Feature shape:", X.shape)
print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training SVM...")
model = SVC(kernel="rbf", C=10, class_weight="balanced")
model.fit(X_train, y_train)

print("Evaluating...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, os.path.join(OUT_DIR, "hog_svm_model.pkl"))
print("Saved model hog_svm_model.pkl")
