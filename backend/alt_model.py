"""
Alternative model loader that uses TensorFlow directly but with extra Windows compatibility.
This provides a more robust way to load and use the model on Windows systems.
"""

import os
import sys
import platform
import numpy as np
from PIL import Image
import logging
import importlib.util
from pathlib import Path

logger = logging.getLogger(__name__)

# Disease classes (must match those in app.py)
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

# Add potential DLL directories for Windows
if platform.system() == 'Windows':
    try:
        # Add Windows system directories to DLL search path
        os.add_dll_directory("C:/Windows/System32")
        os.add_dll_directory("C:/Windows/System32/downlevel")
        
        # Try to detect installed CUDA
        possible_cuda_dirs = [
            Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA"),
            Path(os.path.expanduser("~/.conda/envs/tf/Library/bin")),
            Path("C:/Program Files/NVIDIA Corporation/Nsight Systems"),
            *[Path(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{ver}/bin") 
              for ver in ["12.0", "11.8", "11.6", "11.4", "11.2", "11.0", "10.2", "10.1", "10.0"]]
        ]
        
        # Add all existing CUDA paths to DLL search
        for cuda_dir in possible_cuda_dirs:
            if cuda_dir.exists():
                try:
                    os.add_dll_directory(str(cuda_dir))
                    if (cuda_dir / "bin").exists():
                        os.add_dll_directory(str(cuda_dir / "bin"))
                    logger.info(f"Added CUDA directory: {cuda_dir}")
                except Exception as e:
                    logger.warning(f"Failed to add CUDA directory {cuda_dir}: {e}")
        
        # Look for TensorFlow DLLs in Python path
        python_path = Path(sys.executable).parent
        try:
            if python_path.exists():
                os.add_dll_directory(str(python_path))
                logger.info(f"Added Python directory: {python_path}")
        except Exception as e:
            logger.warning(f"Failed to add Python directory {python_path}: {e}")
            
    except Exception as e:
        logger.warning(f"Error setting up Windows DLL directories: {e}")

# Lazy import TensorFlow to avoid immediate loading
tf = None
keras = None

def load_tensorflow():
    """Load TensorFlow with special handling for Windows."""
    global tf, keras
    
    if tf is not None and keras is not None:
        return True
        
    try:
        # First check if TensorFlow is available without importing
        if importlib.util.find_spec("tensorflow") is None:
            logger.error("TensorFlow not installed")
            return False
            
        # Set environment variable to control TensorFlow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity
        
        # Now attempt to import TensorFlow
        import tensorflow
        # Print TensorFlow version to verify it's loaded
        logger.info(f"Loaded TensorFlow {tensorflow.__version__}")
        
        # Import Keras from tensorflow
        from tensorflow import keras
        
        # Set global variables
        global tf, keras
        tf = tensorflow
        keras = tensorflow.keras
        
        # Configure TensorFlow to grow memory usage
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except Exception as e:
                    logger.warning(f"Failed to set memory growth: {e}")
        
        return True
    except ImportError as e:
        logger.error(f"Error importing TensorFlow: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error loading TensorFlow: {e}")
        return False

# Global model variable with lazy loading
_model = None

def get_model():
    """Lazily load the model only when needed."""
    global _model
    
    if _model is not None:
        return _model
        
    # Check if we can load TensorFlow
    if not load_tensorflow():
        logger.error("Failed to load TensorFlow, model cannot be loaded")
        return None
        
    try:
        # Get model path
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plant_disease_model.h5')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None
            
        # Load model with Keras
        logger.info(f"Loading model from {model_path}")
        _model = keras.models.load_model(model_path, compile=False)
        
        # Verify model output shape
        output_shape = _model.output_shape
        logger.info(f"Model output shape: {output_shape}")
        
        if output_shape[1] != len(DISEASE_CLASSES):
            logger.warning(f"Model output shape ({output_shape[1]}) doesn't match number of disease classes ({len(DISEASE_CLASSES)})")
        
        return _model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def preprocess_image(img):
    """Preprocess an image for model prediction."""
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize to expected size (224x224 for standard plant disease models)
    img = img.resize((224, 224))
    
    # Convert to numpy array and normalize to 0-1
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_model_prediction(image_data):
    """Get disease prediction from the model."""
    try:
        # Ensure TensorFlow is loaded
        if not load_tensorflow():
            logger.error("Cannot make prediction - TensorFlow not available")
            return None, 0.0
        
        # Preprocess the image
        if isinstance(image_data, Image.Image):
            processed_img = preprocess_image(image_data)
        else:
            # Assume it's an already processed numpy array
            processed_img = image_data
        
        # Load the model
        model = get_model()
        if model is None:
            logger.error("Model could not be loaded")
            return None, 0.0
        
        # Make prediction
        logger.info("Running model prediction")
        predictions = model.predict(processed_img)
        
        # Get highest confidence class
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        
        # Get disease name
        disease = DISEASE_CLASSES[predicted_class_index]
        
        logger.info(f"Predicted disease: {disease}, confidence: {confidence:.4f}")
        return disease, confidence
        
    except Exception as e:
        logger.error(f"Error in model prediction: {e}")
        tb_info = sys.exc_info()[2]
        logger.error(f"Line: {tb_info.tb_lineno}")
        return None, 0.0

# Testing
if __name__ == "__main__":
    # Test with a random image
    test_array = np.random.rand(1, 224, 224, 3)
    disease, confidence = get_model_prediction(test_array)
    print(f"Predicted disease: {disease}, confidence: {confidence:.4f}") 