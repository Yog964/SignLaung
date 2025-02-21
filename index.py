from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
import io
import os
import requests  # <-- Import requests to download files
from PIL import Image

app = Flask(__name__)
CORS(app)

# Function to download model files if they don’t exist
def download_model(file_url, file_name):
    """Downloads a model file if not already present."""
    if not os.path.exists(file_name):
        print(f"Downloading {file_name} from {file_url}...")
        response = requests.get(file_url)
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"{file_name} downloaded successfully!")

# URLs for the model files on GitHub
VEG_MODEL_URL = "https://github.com/Yog964/SignLaung/raw/main/Veg2102.pkl"
FRUIT_MODEL_URL = "https://github.com/Yog964/SignLaung/raw/main/Fruit2102.pkl"

# Download models if missing
download_model(VEG_MODEL_URL, "Veg2102.pkl")
download_model(FRUIT_MODEL_URL, "Fruit2102.pkl")

# Load models
vegetable_model = joblib.load("Veg2102.pkl")
fruit_model = joblib.load("Fruit2102.pkl")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_frame(image, model_type):
    """Extract pose landmarks and predict action based on model type."""
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        landmarks = np.array(landmarks).reshape(1, -1)

        # Use the correct model based on model_type
        if model_type == "vegetables":
            action = vegetable_model.predict(landmarks)[0]
        elif model_type == "fruit":
            action = fruit_model.predict(landmarks)[0]
        else:
            action = "Invalid Model Type"
        return action
    return "No Pose Detected"

@app.route('/predict', methods=['POST'])
def predict():
    """Receive frame, process it, and return predicted action."""
    data = request.json
    image_data = data['image']
    model_type = data['model_type']  # Get the model type from the request

    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)

    action = process_frame(image, model_type)

    return jsonify({"action": action})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)
