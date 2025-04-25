import tensorflow as tf
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
import io
from collections import Counter
import os
import logging
from tensorflow.keras.models import load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = None

# Define possible model paths to check
MODEL_PATHS = [
    "models/finetuned_cnn_model.keras",
    "./models/finetuned_cnn_model.keras",
    "/app/models/finetuned_cnn_model.keras",  # Common path in containerized environments
    os.path.join(os.getcwd(), "models/finetuned_cnn_model.keras")
]

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
    """Extract frames from a video stored in memory"""
    # Save the video bytes to a temporary file
    try:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_bytes.getbuffer())
        
        # Open the video file and extract frames
        cap = cv2.VideoCapture("temp_video.mp4")
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return []
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        # Clean up the temporary video file
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
        
        return frames
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
        return []

def preprocess_frame(frame):
    """Preprocess a frame for the model"""
    # Resize the frame to match the model's input size
    frame = cv2.resize(frame, (224, 224))
    
    # Normalize the frame to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Expand dimensions to match the model's input shape
    frame = np.expand_dims(frame, axis=0)
    
    return frame

def find_model_path():
    """Find the actual path to the model file"""
    logger.info("Searching for model file...")
    
    # Log current directory for debugging
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")
    
    # List files in the models directory if it exists
    models_dir = os.path.join(current_dir, "models")
    if os.path.exists(models_dir):
        logger.info(f"Contents of models directory: {os.listdir(models_dir)}")
    else:
        logger.warning(f"Models directory not found at {models_dir}")
    
    # Check all possible paths
    for path in MODEL_PATHS:
        if os.path.exists(path):
            logger.info(f"Model found at: {path}")
            return path
    
    # If we get here, no model path was found
    logger.error("Model file not found in any of the expected locations")
    return None

@app.on_event("startup")
def load_cnn_model():
    """Load the CNN model on application startup"""
    global model
    try:
        # Find the model path
        model_path = find_model_path()
        
        if not model_path:
            logger.error("Model file not found. The API will run but predictions will fail.")
            return
        
        # Load the model
        logger.info(f"Loading Keras model from {model_path}...")
        model = load_model(model_path)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Dockerized FastAPI on Crane Cloud!"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "online",
        "model_status": model_status
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")
            
        # Log the received file
        logger.info(f"Received file: {file.filename}")

        # Read the video file
        contents = await file.read()
        video_bytes = io.BytesIO(contents)

        # Extract frames from the video
        frames = extract_frames(video_bytes)
        logger.info(f"Extracted {len(frames)} frames from the video.")

        if not frames:
            raise HTTPException(status_code=400, detail="No frames extracted from the video.")

        # Process each frame and perform inference
        predictions = []
        for frame in frames:
            # Resize and normalize the frame
            processed_frame = preprocess_frame(frame)
            
            # Perform inference
            prediction = model.predict(processed_frame, verbose=0)  # Suppress verbose output
            predicted_class = np.argmax(prediction, axis=1)[0]
            predictions.append(predicted_class)

        # Log the predicted classes
        logger.info(f"Predicted classes: {predictions}")

        # Use a voting mechanism to determine the final translation
        final_class = Counter(predictions).most_common(1)[0][0]
        translation = class_to_word.get(final_class, "Unknown")
        logger.info(f"Final translation: {translation}")

        return {"translation": translation}
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
