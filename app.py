from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from flask_cors import CORS  # <== important for separate frontend!

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model
model = tf.keras.models.load_model("final_model.h5")
class_names = ["Infected", "Not Infected"]

def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    img = preprocess_image(file)
    preds = model.predict(img)
    result = class_names[np.argmax(preds)]

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
