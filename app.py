import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, request, jsonify
import io
from collections import Counter
import os
import gdown
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Google Drive file ID and model path
MODEL_FILE_ID = "1nI--fH-H5mzGFB8rFpSkx2Ld8zhzewlS"  # Replace with your file ID
MODEL_PATH = "sign_language_model_finetuned.keras"

# Download the model from Google Drive if it doesn't exist
if not os.path.exists(MODEL_PATH):
    logger.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Load the Keras model
logger.info("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)
logger.info("Model loaded successfully!")

# Define the class-to-word mapping
class_to_word = {
    28: "hello",
    84: "you",
    31: "how",
    46: "name",
    78: "what",
    63: "stay",
    79: "where",
    50: "old",
    41: "me",
    8: "can",
    64: "teach",
    67: "thank you",
    29: "help",
    57: "please",
    59: "repeat",
    24: "good",
    44: "morning",
    1: "afternoon",
    18: "evening",
    13: "day",
    5: "bad",
    32: "hungry",
    68: "thirsty",
    25: "goodbye",
    40: "market",
    16: "directions",
    38: "looking",
    6: "book",
    37: "learning",
    65: "teach me",
    23: "game",
    77: "we",
    60: "rules",
    56: "playing",
    0: "about",
    36: "know",
    12: "culture and tradition",
    80: "wish",
    69: "to",
    72: "travel",
    82: "world",
    42: "meet people",
    33: "interact",
    55: "planning",
    75: "visit",
    47: "next",
    3: "any tips",
    83: "would you like",
    35: "join us",
    15: "dinner",
    81: "work",
    66: "team",
    45: "moving",
    22: "forward",
    51: "open the door",
    17: "enter",
    10: "come",
    11: "cook",
    21: "food",
    70: "to eat",
    39: "lunch",
    2: "and",
    4: "as follows",
    7: "but",
    9: "children",
    14: "different",
    19: "family",
    20: "follows",
    26: "guardians",
    27: "have",
    30: "here",
    34: "is",
    43: "members",
    48: "now",
    49: "numbers",
    52: "other",
    53: "our",
    54: "parents",
    58: "regions",
    61: "sign language",
    62: "start",
    71: "topic",
    73: "ugandan",
    74: "users",
    76: "ways"
}

def extract_frames(video_bytes):
    # Save the video bytes to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(video_bytes)
    
    # Open the video file and extract frames
    cap = cv2.VideoCapture("temp_video.mp4")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    # Clean up the temporary video file
    os.remove("temp_video.mp4")
    
    return frames

def preprocess_frame(frame):
    # Resize the frame to match the model's input size
    frame = cv2.resize(frame, (224, 224))
    
    # Normalize the frame to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Expand dimensions to match the model's input shape
    frame = np.expand_dims(frame, axis=0)
    
    return frame

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "Dockerized Flask API on Crane Cloud!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Log the received file
        logger.info(f"Received file: {file.filename}")

        # Read the video file
        contents = file.read()
        
        # Extract frames from the video
        frames = extract_frames(contents)
        logger.info(f"Extracted {len(frames)} frames from the video.")

        # Process each frame and perform inference
        predictions = []
        for frame in frames:
            # Resize and normalize the frame
            processed_frame = preprocess_frame(frame)
            
            # Perform inference
            prediction = model.predict(processed_frame)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predictions.append(predicted_class)

        # Log the predicted classes
        logger.info(f"Predicted classes: {predictions}")

        # Use a voting mechanism to determine the final translation
        if not predictions:
            return jsonify({"error": "No frames extracted from the video."}), 400
        
        final_class = Counter(predictions).most_common(1)[0][0]
        translation = class_to_word.get(final_class, "Unknown")
        logger.info(f"Final translation: {translation}")

        return jsonify({"translation": translation})
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
