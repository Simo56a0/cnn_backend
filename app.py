import tensorflow as tf
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from collections import Counter
import os
import gdown
from ai_edge_litert import Interpreter

app = FastAPI()

# Google Drive file ID and model path
MODEL_FILE_ID = "1_Yi_pMGwMDjklGAeEl-IsYVDpiROvYjZ"  # Replace with your file ID
MODEL_PATH = "sign_language_model.tflite"

# Download the model from Google Drive if it doesn't exist
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Load the TensorFlow Lite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the class-to-word mapping
class_to_word = {
    0: "Hello",
    1: "Thank You",
    2: "Yes",
    3: "No"
}

@app.get("/")
async def root():
    return {"message": "Welcome to the Sign Language Translator API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the video file
    contents = await file.read()
    video_bytes = io.BytesIO(contents)
    
    # Extract frames from the video
    frames = extract_frames(video_bytes)
    
    # Process each frame and perform inference
    predictions = []
    for frame in frames:
        # Resize and normalize the frame
        processed_frame = preprocess_frame(frame)
        
        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], processed_frame)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction, axis=1)[0]
        predictions.append(predicted_class)
    
    # Use a voting mechanism to determine the final translation
    final_class = Counter(predictions).most_common(1)[0][0]
    translation = class_to_word.get(final_class, "Unknown")
    
    return {"translation": translation}

def extract_frames(video_bytes):
    # Save the video bytes to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(video_bytes.getbuffer())
    
    # Open the video file and extract frames
    cap = cv2.VideoCapture("temp_video.mp4")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def preprocess_frame(frame):
    # Resize the frame to match the model's input size
    frame = cv2.resize(frame, (224, 224))
    
    # Normalize the frame
    frame = frame / 255.0
    
    # Expand dimensions to match the model's input shape
    frame = np.expand_dims(frame, axis=0)
    frame = frame.astype(np.float32)  # Ensure the input is float32
    return frame
