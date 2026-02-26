import os
import numpy as np
import tensorflow as tf
import cv2
import shutil
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ==============================
# Flask Setup
# ==============================
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
RESULT_FOLDER = os.path.join(STATIC_FOLDER, 'results')

MODEL_PATH = r'C:\Users\usman\OneDrive\Desktop\Batch10\model\best_model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ==============================
# Load Model
# ==============================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully.")


# ==============================
# Utility
# ==============================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==============================
# 🎯 Class-Specific Grad-CAM
# ==============================
def generate_gradcam(img_path, model, target_class, last_conv_layer_name="conv5_block3_out"):

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_exp = np.expand_dims(img_array, axis=0)
    img_array_exp = preprocess_input(img_array_exp)

    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute Gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array_exp)

        # ✅ FIX: handle list output
        if isinstance(predictions, list):
            predictions = predictions[0]

        predictions = tf.convert_to_tensor(predictions)

        # Binary classifier (sigmoid output)
        if target_class == 1:   # Pneumonia
            loss = predictions[:, 0]
        else:                   # Normal
            loss = 1 - predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize for better visualization
    original = cv2.imread(img_path)
    original = cv2.resize(original, (512, 512))

    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)

    # Smooth for clean visualization
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)

    # Apply different color maps
    if target_class == 1:
        colored_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)     # Red/Yellow
        alpha = 0.45
    else:
        colored_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_OCEAN)  # Blue/Green
        alpha = 0.35

    overlay = cv2.addWeighted(original, 1 - alpha, colored_map, alpha, 0)

    return overlay


# ==============================
# Routes
# ==============================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/about')
def about_page():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # =============================
        # Preprocess
        # =============================
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_exp = np.expand_dims(img_array, axis=0)
        img_array_exp = preprocess_input(img_array_exp)

        # =============================
        # Prediction
        # =============================
        pred = model.predict(img_array_exp)

        if isinstance(pred, list):
            pred = pred[0]

        prob_pneumonia = float(np.squeeze(pred))
        prob_normal = 1 - prob_pneumonia

        THRESHOLD = 0.45

        if prob_pneumonia >= THRESHOLD:
            label = "PNEUMONIA"
            confidence = prob_pneumonia
            target_class = 1
        else:
            label = "NORMAL"
            confidence = prob_normal
            target_class = 0

        # =============================
        # Grad-CAM
        # =============================
        overlay = generate_gradcam(filepath, model, target_class)

        cam_filename = f"cam_{filename}"
        cam_path = os.path.join(RESULT_FOLDER, cam_filename)
        cv2.imwrite(cam_path, overlay)

        original_static_path = os.path.join(RESULT_FOLDER, f"orig_{filename}")
        shutil.copy(filepath, original_static_path)

        return render_template(
            'result.html',
            label=label,
            confidence=round(confidence * 100, 2),
            original_image=f"results/orig_{filename}",
            gradcam_image=f"results/{cam_filename}"
        )

    return redirect(request.url)


# ==============================
# Run App
# ==============================
if __name__ == '__main__':
    app.run(debug=True, port=5000)