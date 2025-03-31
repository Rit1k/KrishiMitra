import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import sys

# Configure basic logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define DISEASE_CLASSES - same as in app.py
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def preprocess_image(img):
    logger.info("Preprocessing image")
    # Resize image to match model's expected sizing
    img = img.resize((224, 224))
    # Convert to array and normalize
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    logger.info("Image preprocessing successful")
    return x

def load_and_test_model():
    try:
        # Load the model
        model_path = 'plant_disease_model.h5'
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at: {model_path}")
            return False
            
        logger.info(f"Loading model from: {model_path}")
        model = load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Create a simple test with random data
        logger.info("Creating test data")
        test_input = np.random.rand(1, 224, 224, 3)
        
        # Perform a test prediction
        logger.info("Running test prediction")
        predictions = model.predict(test_input)
        
        # Check the output shape
        logger.info(f"Prediction output shape: {predictions.shape}")
        logger.info(f"Number of classes in output: {predictions.shape[1]}")
        logger.info(f"Number of classes in DISEASE_CLASSES: {len(DISEASE_CLASSES)}")
        
        # Check if number of output classes matches our disease classes list
        if predictions.shape[1] == len(DISEASE_CLASSES):
            logger.info("✅ Model output classes match our disease classes list")
        else:
            logger.warning(f"⚠️ Model output has {predictions.shape[1]} classes but DISEASE_CLASSES has {len(DISEASE_CLASSES)}")
        
        # Sample prediction
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        logger.info(f"Sample prediction: Class {predicted_class_index} - {DISEASE_CLASSES[predicted_class_index]} with confidence {confidence}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting model test")
    success = load_and_test_model()
    if success:
        logger.info("Model test completed successfully")
        sys.exit(0)
    else:
        logger.error("Model test failed")
        sys.exit(1) 