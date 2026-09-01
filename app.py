from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

from tensorflow.keras import layers, models as keras_models

def build_original_cnn():
    return keras_models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

model = build_original_cnn()
model.load_weights("clothing_model.weights.h5")

with open("class_names.json") as f:
    class_names = json.load(f)

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    top_idx = int(np.argmax(predictions))
    top_class = class_names[top_idx]
    top_confidence = float(predictions[top_idx] * 100)

    top3_idx = np.argsort(predictions)[-3:][::-1]
    top3 = [(class_names[i], float(predictions[i] * 100)) for i in top3_idx]

    return top_class, top_confidence, top3

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    top3 = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)
            prediction, confidence, top3 = predict_image(image_path)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        top3=top3,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)