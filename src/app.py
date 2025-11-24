# src/app.py
import os
import logging
from datetime import datetime
from flask import (
    Flask, render_template, request, redirect,
    url_for, send_from_directory, flash, abort, jsonify
)
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from utils import load_class_names

# --- Cấu hình đường dẫn dự án ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "resnet_features_model.h5")
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "class_names_resnet.json")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Flask app ---
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static")
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "very-secret-key")

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load models safely ---
classifier = None
resnet = None
class_names = []

try:
    logger.info("Loading feature classifier model from %s ...", MODEL_PATH)
    classifier = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Classifier loaded.")
except Exception as e:
    logger.exception("Failed to load classifier model: %s", e)
    # Do not exit; app will still start but predict attempts will return errors.

try:
    logger.info("Loading ResNet50 backbone ...")
    resnet = tf.keras.applications.ResNet50(
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3)
    )
    resnet.trainable = False
    logger.info("ResNet50 loaded.")
except Exception as e:
    logger.exception("Failed to load ResNet50 backbone: %s", e)

try:
    class_names = load_class_names(CLASS_NAMES_PATH)
    logger.info("Loaded class count: %d", len(class_names))
except Exception as e:
    logger.exception("Failed to load class names: %s", e)
    class_names = []

# --- Helpers ---
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def extract_resnet_feature(img_path: str) -> np.ndarray:
    """Load image and extract 2048-dim feature using resnet backbone."""
    if resnet is None:
        raise RuntimeError("ResNet model is not loaded.")
    # Keras image utilities
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    feat = resnet.predict(x, verbose=0)  # shape (1, 2048)
    return feat

def predict_from_feature(feature: np.ndarray):
    """Return preds array from classifier given extracted feature."""
    if classifier is None:
        raise RuntimeError("Classifier model is not loaded.")
    preds = classifier.predict(feature, verbose=0)  # expect shape (1, num_classes)
    return preds

def template_exists(name: str) -> bool:
    return os.path.exists(os.path.join(app.template_folder, name))

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Nếu có template 'index.html' sẽ render template với biến 'result'.
    Nếu không, trả về HTML đơn giản / JSON.
    """
    if request.method == "POST":
        # Basic checks
        if "file" not in request.files:
            flash("Không tìm thấy file upload")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("Bạn chưa chọn file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Unique filename to avoid collisions
            orig_filename = secure_filename(file.filename)
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            filename = f"{timestamp}_{orig_filename}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            try:
                file.save(save_path)
            except Exception as e:
                logger.exception("Lỗi khi lưu file: %s", e)
                flash("Lỗi khi lưu file lên server.")
                return redirect(request.url)

            # Try extracting features and predicting
            try:
                feature = extract_resnet_feature(save_path)  # (1,2048)
            except Exception as e:
                logger.exception("Lỗi khi trích xuất feature: %s", e)
                flash("Lỗi khi xử lý ảnh (extract feature).")
                return redirect(request.url)

            try:
                preds = predict_from_feature(feature)[0]  # shape (num_classes,)
            except Exception as e:
                logger.exception("Lỗi khi dự đoán (classifier): %s", e)
                flash("Lỗi khi dự đoán. Kiểm tra model classifier.")
                return redirect(request.url)

            # Validate preds length vs class_names
            if len(class_names) != 0 and len(preds) != len(class_names):
                logger.warning("Số lượng preds (%d) != số lớp (%d).", len(preds), len(class_names))

            # Top-1
            top_idx = int(np.argmax(preds))
            top_name = class_names[top_idx] if 0 <= top_idx < len(class_names) else str(top_idx)
            top_prob = float(preds[top_idx])

            # Top-5
            top5_idx = np.argsort(preds)[::-1][:5]
            top5 = [
                (class_names[int(i)] if 0 <= int(i) < len(class_names) else str(int(i)),
                 float(preds[int(i)]))
                for i in top5_idx
            ]

            result = {
                "filename": filename,
                "file_url": url_for("uploaded_file", filename=filename),
                "top1": {"class": top_name, "prob": top_prob},
                "top5": [{"class": name, "prob": prob} for name, prob in top5]
            }

            # Nếu có template, render; nếu không trả JSON đơn giản
            if template_exists("index.html"):
                return render_template("index.html", result=result)
            else:
                # Simple HTML fallback
                html = f"""
                <h2>Kết quả dự đoán</h2>
                <p><strong>Top-1:</strong> {result['top1']['class']} ({result['top1']['prob']:.4f})</p>
                <p><strong>Ảnh:</strong> <a href="{result['file_url']}">Xem ảnh upload</a></p>
                <h3>Top-5</h3>
                <ol>
                """
                for e in result["top5"]:
                    html += f"<li>{e['class']} ({e['prob']:.4f})</li>"
                html += "</ol>"
                html += '<p><a href="/">Quay lại</a></p>'
                return html

        else:
            flash("Định dạng file không được phép. Chỉ chấp nhận png/jpg/jpeg.")
            return redirect(request.url)

    # GET
    if template_exists("index.html"):
        return render_template("index.html", result=None)
    else:
        # Simple upload form fallback
        return """
        <h2>Upload image for prediction</h2>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required>
            <br><br>
            <input type="submit" value="Upload & Predict">
        </form>
        """

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    # Serve uploaded file
    safe_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(safe_path):
        abort(404)
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/predict_json", methods=["POST"])
def predict_json():
    """
    API endpoint: nhận multipart/form-data 'file' hoặc JSON base64 (nếu muốn mở rộng).
    Trả JSON với top1/top5.
    """
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "invalid file extension"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        feature = extract_resnet_feature(save_path)
        preds = predict_from_feature(feature)[0]
    except Exception as e:
        logger.exception("Error in predict_json: %s", e)
        return jsonify({"error": "prediction failed", "detail": str(e)}), 500

    top_idx = int(np.argmax(preds))
    top_name = class_names[top_idx] if 0 <= top_idx < len(class_names) else str(top_idx)
    top_prob = float(preds[top_idx])
    top5_idx = np.argsort(preds)[::-1][:5]
    top5 = [
        {"class": class_names[int(i)] if 0 <= int(i) < len(class_names) else str(int(i)),
         "prob": float(preds[int(i)])}
        for i in top5_idx
    ]

    return jsonify({
        "filename": filename,
        "file_url": url_for("uploaded_file", filename=filename),
        "top1": {"class": top_name, "prob": top_prob},
        "top5": top5
    })

# --- Error handlers ---
@app.errorhandler(413)
def request_entity_too_large(error):
    flash("File quá lớn (hạn mức 8 MB).")
    return redirect(url_for("index"))

@app.errorhandler(404)
def not_found(e):
    return "Không tìm thấy.", 404

# --- Run app ---
if __name__ == "__main__":
    # chạy debug cho dev; chuyển đổi theo môi trường khi deploy
    app.run(host="0.0.0.0", port=5000, debug=True)
