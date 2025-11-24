# src/eda.py

import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Đường dẫn dataset
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "flower_data")

TRAIN_DIR = os.path.join(DATA_ROOT, "train")


def sample_head(folder, n=5):
    """Lấy vài ảnh đầu tiên trong folder."""
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
    return sorted(files)[:n]


def show_samples_per_class(base_dir=TRAIN_DIR, n_classes=5, n_per_class=3):
    """
    Hiển thị ảnh mẫu của một số class.
    Tránh hiển thị 102 class gây lỗi kéo dài figure.
    """
    classes = sorted([d for d in os.listdir(base_dir)
                      if os.path.isdir(os.path.join(base_dir, d))])

    # Lấy ngẫu nhiên 5 class
    classes = random.sample(classes, min(len(classes), n_classes))

    samples = []
    for c in classes:
        class_dir = os.path.join(base_dir, c)
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith((".jpg",".jpeg",".png"))]

        chosen = random.sample(files, min(len(files), n_per_class))

        for file in chosen:
            samples.append((c, os.path.join(class_dir, file)))

    total = len(samples)
    cols = n_per_class
    rows = n_classes

    plt.figure(figsize=(cols * 4, rows * 4))

    for i, (cls, path) in enumerate(samples):
        img = Image.open(path).convert("RGB")
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"Class {cls}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def class_distribution(train_dir=TRAIN_DIR):
    """Đếm số lượng ảnh trong từng class."""
    classes = sorted([d for d in os.listdir(train_dir)
                      if os.path.isdir(os.path.join(train_dir, d))])
    counts = []

    for c in classes:
        folder = os.path.join(train_dir, c)
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg",".jpeg",".png"))]
        counts.append((c, len(files)))

    df = pd.DataFrame(counts, columns=["class", "count"]).sort_values("count", ascending=False)
    return df


if __name__ == "__main__":
    print("📌 PHÂN PHỐI SỐ LƯỢNG ẢNH TRONG TỪNG CLASS:")
    df = class_distribution()
    print(df.head(20).to_string(index=False))

    print("\n📌 HIỂN THỊ ẢNH MẪU:")
    show_samples_per_class()
