from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS         
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging
import logging.handlers
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import sys
import traceback
import requests
import json
from datetime import datetime
import numpy as np
import io
from PIL import Image
import platform
import google.generativeai as genai

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging with rotation
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = os.path.join('logs', 'krishimitra.log')
file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)

# Set up console handler with color formatting
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Configure root logger
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)

# Capture all unhandled exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Don't capture keyboard interrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# Load environment variables
load_dotenv()

# Initialize default variables
TF_AVAILABLE = False
model = None
GENAI_AVAILABLE = False
OPENAI_AVAILABLE = False
TWILIO_AVAILABLE = False
ALT_PREDICTOR_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Before the TensorFlow import, add a fix for Windows DLL loading issues
if platform.system() == 'Windows':
    try:
        # Use the same robust Windows-specific path handling from our alt_model module
        # This ensures DLLs can be found on Windows systems
        from alt_model import load_tensorflow
        # Import and use TensorFlow through our helper function
        tf_loaded = load_tensorflow()
        if tf_loaded:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing import image
            logger.info("TensorFlow successfully imported via alt_model helper")
            TF_AVAILABLE = True
        else:
            logger.warning("Failed to load TensorFlow via alt_model helper")
            TF_AVAILABLE = False
            model = None
    except Exception as e:
        logger.error(f"Error using alt_model to load TensorFlow: {str(e)}")
        # Fall back to regular import approach
        try:
            # Add standard Windows system directories to DLL search path
            os.add_dll_directory("C:/Windows/System32")
            os.add_dll_directory("C:/Windows/System32/downlevel")
            
            # Try to find CUDA DLLs if they exist
            cuda_paths = [
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin"
            ]
            
            for path in cuda_paths:
                if os.path.exists(path):
                    try:
                        os.add_dll_directory(path)
                        logger.info(f"Added CUDA DLL directory: {path}")
                    except Exception as e:
                        logger.warning(f"Failed to add CUDA DLL directory {path}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error setting up DLL directories: {str(e)}")
else:
    # Non-Windows platforms don't need special DLL handling
    pass

# Try to import TensorFlow and set up the model
if not (platform.system() == 'Windows' and 'tf' in locals() and TF_AVAILABLE):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        
        # Fix TensorFlow memory issues
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        # Set TensorFlow logging level to suppress warnings
        tf.get_logger().setLevel('ERROR')
        
        TF_AVAILABLE = True
        logger.info("TensorFlow successfully imported")
    except ImportError:
        logger.warning("TensorFlow could not be imported - disease detection will be unavailable")
        TF_AVAILABLE = False
        model = None
    except Exception as e:
        logger.error(f"Error initializing TensorFlow: {str(e)}")
        TF_AVAILABLE = False
        model = None

# Try to load the disease detection model
if TF_AVAILABLE:
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plant_disease_model.h5')
        if os.path.exists(model_path):
            logger.info(f"Loading model from: {model_path}")
            model = load_model(model_path, compile=False)  # Don't compile the model to avoid additional issues
            logger.info("Model loaded successfully")
            logger.info(f"Model output shape: {model.output_shape}")
            logger.info(f"Number of disease classes: {len(DISEASE_CLASSES)}")
            
            # Check if model output matches our disease classes
            if model.output_shape[1] == len(DISEASE_CLASSES):
                logger.info("Model output shape matches number of disease classes")
            else:
                logger.warning(f"Model output shape ({model.output_shape[1]}) doesn't match number of disease classes ({len(DISEASE_CLASSES)})")
        else:
            logger.warning(f"Model file not found at: {model_path}")
            model = None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None

# Try to import the alternative model predictor
ALT_PREDICTOR_AVAILABLE = False
try:
    import alt_model
    ALT_PREDICTOR_AVAILABLE = True
    logger.info("Alternative model predictor is available")
except ImportError:
    logger.warning("Alternative model predictor not available")
except Exception as e:
    logger.error(f"Error initializing alternative model predictor: {str(e)}")

# Try to import and configure other APIs - also made optional
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
    logger.info("Google GenerativeAI successfully imported")
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("Google GenerativeAI could not be imported - chat functionality will be limited")

try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI successfully imported")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI could not be imported - may limit fallback functionality")

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
    logger.info("Twilio successfully imported")
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio could not be imported - SMS functionality will be unavailable")

# API Keys and Configurations
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
WEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

# Configure Gemini API
if GEMINI_API_KEY:
    try:
        logger.info(f"Attempting to configure Gemini API with key: {GEMINI_API_KEY[:10]}...")
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured successfully")
        
        # Test the API configuration
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content("Test connection")
            logger.info("Gemini API test successful")
        except Exception as test_error:
            logger.error(f"Gemini API test failed: {str(test_error)}")
            GEMINI_API_KEY = None
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {str(e)}")
        GEMINI_API_KEY = None
else:
    logger.warning("Gemini API key not found in environment variables")

# Log available API keys (without exposing the actual keys)
logger.info(f"OpenAI API key available: {bool(OPENAI_API_KEY)}")
logger.info(f"Gemini API key available: {bool(GEMINI_API_KEY)}")
logger.info(f"OpenWeather API key available: {bool(WEATHER_API_KEY)}")
logger.info(f"Twilio account SID available: {bool(TWILIO_ACCOUNT_SID)}")
logger.info(f"Twilio auth token available: {bool(TWILIO_AUTH_TOKEN)}")
logger.info(f"Google Maps API key available: {bool(GOOGLE_MAPS_API_KEY)}")

# Initialize APIs
twilio_client = None

# Initialize Twilio if available
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_AVAILABLE:
    try:
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        logger.info("Twilio client initialized")
    except Exception as e:
        logger.error(f"Error initializing Twilio client: {str(e)}")

# Disease classes (update these according to your model's classes)
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

# Disease treatments (you can expand this dictionary)
DISEASE_TREATMENTS = {
    'Apple___Apple_scab': 'Apply fungicides containing captan or myclobutanil. Remove infected leaves and maintain good air circulation.',
    'Apple___Black_rot': 'Apply fungicides with captan or thiophanate-methyl. Remove infected fruit and prune out diseased branches and cankers.',
    'Apple___Cedar_apple_rust': 'Apply fungicides containing myclobutanil or propiconazole. Remove nearby cedar trees if possible to break the disease cycle.',
    'Apple___healthy': 'No treatment needed. Maintain good cultural practices including proper pruning, adequate nutrition, and regular monitoring.',
    'Blueberry___healthy': 'No treatment needed. Maintain good cultural practices including proper soil pH (4.5-5.5), adequate mulching, and regular monitoring.',
    'Cherry_(including_sour)___Powdery_mildew': 'Apply sulfur-based fungicides or potassium bicarbonate. Prune to improve air circulation and avoid overhead irrigation.',
    'Cherry_(including_sour)___healthy': 'No treatment needed. Maintain good cultural practices including proper pruning, adequate nutrition, and regular monitoring.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply fungicides containing pyraclostrobin or azoxystrobin. Practice crop rotation and consider resistant varieties.',
    'Corn_(maize)___Common_rust': 'Apply fungicides containing azoxystrobin or propiconazole. Plant resistant hybrids and time planting to avoid peak rust periods.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Apply fungicides with pyraclostrobin or propiconazole. Plant resistant hybrids and implement crop rotation.',
    'Corn_(maize)___healthy': 'No treatment needed. Maintain good cultural practices including crop rotation, proper fertility, and weed management.',
    'Grape___Black_rot': 'Apply fungicides containing myclobutanil or mancozeb. Remove mummified berries and prune out infected wood.',
    'Grape___Esca_(Black_Measles)': 'No effective chemical treatment. Prune in dry weather, disinfect tools, and apply wound sealant. Severely infected vines may need removal.',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply fungicides with mancozeb or copper compounds. Improve air circulation and avoid overhead irrigation.',
    'Grape___healthy': 'No treatment needed. Maintain good cultural practices including proper pruning, canopy management, and regular monitoring.',
    'Orange___Haunglongbing_(Citrus_greening)': 'No cure available. Control Asian citrus psyllid with insecticides. Remove and destroy infected trees to prevent spread.',
    'Peach___Bacterial_spot': 'Apply copper-based bactericides. Plant resistant varieties and maintain good air circulation through proper pruning.',
    'Peach___healthy': 'No treatment needed. Maintain good cultural practices including proper pruning, adequate nutrition, and regular monitoring.',
    'Pepper,_bell___Bacterial_spot': 'Apply copper-based bactericides. Use pathogen-free seeds and practice crop rotation. Avoid overhead irrigation.',
    'Pepper,_bell___healthy': 'No treatment needed. Maintain good cultural practices including proper spacing, adequate nutrition, and regular monitoring.',
    'Potato___Early_blight': 'Apply fungicides containing chlorothalonil or azoxystrobin. Practice crop rotation and provide adequate plant spacing.',
    'Potato___Late_blight': 'Apply fungicides containing chlorothalonil, mancozeb, or metalaxyl. Destroy volunteer plants and ensure good air circulation.',
    'Potato___healthy': 'No treatment needed. Maintain good cultural practices including proper hilling, adequate nutrition, and regular monitoring.',
    'Raspberry___healthy': 'No treatment needed. Maintain good cultural practices including proper pruning, adequate spacing, and regular monitoring.',
    'Soybean___healthy': 'No treatment needed. Maintain good cultural practices including crop rotation, proper fertility, and weed management.',
    'Squash___Powdery_mildew': 'Apply fungicides containing sulfur or potassium bicarbonate. Space plants for good air circulation and avoid overhead irrigation.',
    'Strawberry___Leaf_scorch': 'Apply fungicides containing captan or myclobutanil. Renovate beds annually and provide adequate spacing between plants.',
    'Strawberry___healthy': 'No treatment needed. Maintain good cultural practices including proper spacing, mulching, and regular monitoring.',
    'Tomato___Bacterial_spot': 'Apply copper-based bactericides. Use pathogen-free seeds and practice crop rotation. Remove and destroy infected plant material.',
    'Tomato___Early_blight': 'Apply fungicides containing chlorothalonil or mancozeb. Remove lower leaves showing symptoms and mulch soil to prevent splash-up.',
    'Tomato___Late_blight': 'Apply fungicides containing chlorothalonil, mancozeb, or copper compounds. Improve air circulation and use drip irrigation.',
    'Tomato___Leaf_Mold': 'Apply fungicides containing chlorothalonil or copper compounds. Reduce humidity and improve air circulation. Avoid leaf wetness.',
    'Tomato___Septoria_leaf_spot': 'Apply fungicides containing chlorothalonil or copper compounds. Remove infected leaves promptly and apply mulch to prevent soil splash.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Apply insecticidal soap or horticultural oil. For severe infestations, use miticides. Keep plants well-watered to prevent stress.',
    'Tomato___Target_Spot': 'Apply fungicides containing chlorothalonil or azoxystrobin. Improve air circulation and remove infected plant material.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'No cure once infected. Remove infected plants and control whitefly vectors with appropriate insecticides.',
    'Tomato___Tomato_mosaic_virus': 'No cure once infected. Remove and destroy infected plants. Use virus-free certified seed and disinfect tools between plants.',
    'Tomato___healthy': 'No treatment needed. Maintain good cultural practices including proper staking, pruning, and watering at the base of plants.'
}

def preprocess_image(img):
    logger.debug("Preprocessing image")
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize image to match model's expected sizing (usually 224x224 for plant disease models)
        img = img.resize((224, 224))
        
        # Convert to array and normalize to range 0-1
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize to 0-1 range
        
        logger.debug(f"Image preprocessing complete, shape: {x.shape}")
        return x
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/api/detect-disease', methods=['POST'])
def detect_disease():
    if 'image' not in request.files:
        logger.error("No image file in request")
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read the image
        img = Image.open(io.BytesIO(file.read()))
        
        # Preprocess the image (same preprocessing for both approaches)
        processed_image = preprocess_image(img)
        
        disease = None
        confidence = 0
        using_fallback = False
        
        # Check if TensorFlow and model are available
        if TF_AVAILABLE and model is not None:
            logger.info("Using TensorFlow model for prediction")
            
            try:
                # Make prediction using direct TensorFlow model
                logger.debug("Running model prediction")
                predictions = model.predict(processed_image)
                
                # Get the highest confidence class
                predicted_class_index = np.argmax(predictions[0])
                logger.info(f" predicted_class_index: {predicted_class_index}")
                confidence = float(predictions[0][predicted_class_index])
                
                # Get the disease name from our list, using exact naming from the dataset
                disease = DISEASE_CLASSES[predicted_class_index]
                
                logger.info(f"TensorFlow prediction: {disease} (confidence: {confidence:.4f})")
            except Exception as e:
                logger.error(f"Error during TensorFlow prediction: {str(e)}")
                disease = None
        
        # If TensorFlow approach failed, try alternative method
        if disease is None and ALT_PREDICTOR_AVAILABLE:
            logger.info("Using alternative model predictor")
            
            # Use the alternative predictor - which now uses TensorFlow in a different way
            try:
                alt_disease, alt_confidence = alt_model.get_model_prediction(processed_image)
                
                if alt_disease:
                    disease = alt_disease
                    confidence = alt_confidence
                    logger.info(f"Alternative prediction: {disease} (confidence: {confidence:.4f})")
                else:
                    logger.warning("Alternative predictor returned no result")
            except Exception as e:
                logger.error(f"Error using alternative predictor: {str(e)}")
        
        # If all model approaches failed, use basic fallback
        if disease is None:
            logger.warning("All prediction methods failed, using basic fallback approach")
            using_fallback = True
            
            # Use a simple color-based approach as fallback
            pixels = np.array(img)
            avg_color = pixels.mean(axis=(0, 1))
            
            red_intensity = avg_color[0]
           
        treatment = DISEASE_TREATMENTS.get(disease, 'No specific treatment information available.')
        
        # Add note if using fallback mode
        
        
        return jsonify({
            'disease': disease,
            'confidence': confidence,
            'treatment': treatment,
            'fallback_mode': using_fallback
        })
    
    except Exception as e:
        logger.error(f"Error in disease detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/weather', methods=['GET'])
def get_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}'
    response = requests.get(url)
    return jsonify(response.json())

@app.route('/api/transporters', methods=['GET'])
def get_transporters():
    try:
        lat = request.args.get('lat')
        lon = request.args.get('lon')
        
        if not lat or not lon:
            return jsonify({'error': 'Latitude and longitude are required'}), 400

        # Search for various transport-related places
        transport_types = [
            'moving_company',
            'storage',
            'truck_rental',
            'transit_station',
            'warehouse'
        ]
        
        all_results = []
        
        for place_type in transport_types:
            url = (
                f'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
                f'?location={lat},{lon}'
                f'&radius=10000'  # 10km radius
                f'&type={place_type}'
                f'&key={GOOGLE_MAPS_API_KEY}'
            )
            
            response = requests.get(url)
            data = response.json()
            
            if data.get('status') == 'OK':
                all_results.extend(data.get('results', []))
            elif data.get('status') == 'ZERO_RESULTS':
                continue
            else:
                logger.error(f"Error from Google Places API: {data.get('status')} - {data.get('error_message', 'No error message')}")
                if data.get('status') == 'REQUEST_DENIED':
                    return jsonify({'error': 'API key is invalid or missing'}), 401
                elif data.get('status') == 'OVER_QUERY_LIMIT':
                    return jsonify({'error': 'API quota exceeded'}), 429
        
        # Remove duplicates based on place_id
        seen_places = set()
        unique_results = []
        for place in all_results:
            if place['place_id'] not in seen_places:
                seen_places.add(place['place_id'])
                unique_results.append(place)
        
        return jsonify({
            'results': unique_results,
            'count': len(unique_results)
        })
    
    except requests.RequestException as e:
        logger.error(f"Network error while fetching transport services: {str(e)}")
        return jsonify({'error': 'Failed to fetch transport services'}), 503
    except Exception as e:
        logger.error(f"Unexpected error in get_transporters: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        user_id = data.get('user_id', 'anonymous')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Check subscription status for message limits
        user = None
        if user_id != 'anonymous' and db:
            user = db.users.find_one({'_id': user_id})
            logger.info(f"User found: {user is not None}, User ID: {user_id}")
        
        # For non-subscribers, check message count
        if db and not (user and user.get('subscribed', False)):
            message_count = db.messages.count_documents({'user_id': user_id})
            logger.info(f"Message count for user {user_id}: {message_count}")
            
            # Increase the free message limit to 10 messages
            if message_count >= 10:
                return jsonify({
                    'error': 'Message limit reached',
                    'response': "You've reached the limit of 10 free messages. To continue chatting, please subscribe to our service. You can subscribe by providing your email or phone number in the subscription form.",
                    'limit_reached': True,
                    'message_count': message_count
                }), 403
        
        # Check if Gemini API is properly configured
        if not GEMINI_API_KEY:
            logger.error("Gemini API key not configured or invalid")
            return jsonify({
                'error': 'API configuration error',
                'response': "I'm having trouble with my AI configuration. Please try again in a few minutes."
            }), 503
        
        try:
            # Initialize Gemini AI
            model = genai.GenerativeModel('gemini-pro')
            
            # Prepare agricultural context
            agricultural_prompt = f"""You are KrishiMitra, an agricultural assistant for Indian farmers. 
            Your goal is to provide helpful, accurate advice about farming in India.
            Focus on sustainable practices, local crop varieties, and practical solutions.
            Only provide information that would be feasible for farmers in India.
            
            If you don't know something, admit it and suggest where they might find more information.
            
            User query: {user_message}"""
            
            # Generate response with safety settings
            response = model.generate_content(
                agricultural_prompt,
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            )
            
            if not response or not response.text:
                logger.error("Empty response from Gemini API")
                return jsonify({
                    'error': 'Empty response',
                    'response': "I'm having trouble generating a response. Please try again."
                }), 500
            
            ai_response = response.text
            
            # Log the conversation to database if available
            log_conversation(user_id, user_message, ai_response)
            
            return jsonify({
                'response': ai_response,
                'message_count': message_count if 'message_count' in locals() else 0
            })
            
        except Exception as ai_error:
            logger.error(f"Error with Gemini AI: {str(ai_error)}")
            return jsonify({
                'error': 'AI processing error',
                'response': "I'm having trouble processing your request. Please try again."
            }), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'response': "I'm sorry, something went wrong. Please try again later."
        }), 500

def generate_fallback_response(user_message, user_id):
    # Simple rule-based fallback responses
    fallback_responses = [
        "I'm sorry, I'm having trouble connecting to my knowledge base. Please try again later.",
        "For crop diseases, please use our Disease Detection feature for better assistance.",
        "For weather information, you can check our Weather section to get real-time data.",
        "For transportation needs, please see our Transport section."
    ]
    
    fallback_message = fallback_responses[0]
    if "disease" in user_message.lower():
        fallback_message = fallback_responses[1]
    elif "weather" in user_message.lower():
        fallback_message = fallback_responses[2]
    elif "transport" in user_message.lower():
        fallback_message = fallback_responses[3]
    
    # Log the conversation to database if available
    log_conversation(user_id, user_message, fallback_message)
    
    return jsonify({
        'response': fallback_message,
        'warning': 'Using fallback response system'
    })

def log_conversation(user_id, user_message, ai_response):
    # Log the conversation to database if available
    try:
        if db:
            db.messages.insert_one({
                'user_id': user_id,
                'message': user_message,
                'response': ai_response,
                'timestamp': datetime.utcnow()
            })
    except Exception as db_error:
        logger.error(f"Error logging chat message to database: {str(db_error)}")
        # Continue even if logging fails

@app.route('/api/subscribers/sms', methods=['GET'])
def get_sms_subscribers():
    try:
        subscribers = list(db.users.find({'phone': {'$exists': True}}, {'phone': 1, '_id': 0}))
        return jsonify({'subscribers': subscribers})
    except Exception as e:
        logger.error(f"Error fetching SMS subscribers: {str(e)}")
        return jsonify({'error': 'Failed to fetch subscribers'}), 500

@app.route('/api/subscribers/sms', methods=['DELETE'])
def delete_sms_subscriber():
    try:
        phone = request.json.get('phone')
        if not phone:
            return jsonify({'error': 'Phone number is required'}), 400
            
        result = db.users.update_one(
            {'phone': phone},
            {'$set': {'subscribed': False}}
        )
        
        if result.modified_count > 0:
            return jsonify({'success': True})
        return jsonify({'error': 'Subscriber not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting SMS subscriber: {str(e)}")
        return jsonify({'error': 'Failed to delete subscriber'}), 500

@app.route('/api/subscribers/email', methods=['GET'])
def get_email_subscribers():
    try:
        subscribers = list(db.users.find({'email': {'$exists': True}}, {'email': 1, '_id': 0}))
        return jsonify({'subscribers': subscribers})
    except Exception as e:
        logger.error(f"Error fetching email subscribers: {str(e)}")
        return jsonify({'error': 'Failed to fetch subscribers'}), 500

@app.route('/api/subscribers/email', methods=['DELETE'])
def delete_email_subscriber():
    try:
        email = request.json.get('email')
        if not email:
            return jsonify({'error': 'Email is required'}), 400
            
        result = db.users.update_one(
            {'email': email},
            {'$set': {'subscribed': False}}
        )
        
        if result.modified_count > 0:
            return jsonify({'success': True})
        return jsonify({'error': 'Subscriber not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting email subscriber: {str(e)}")
        return jsonify({'error': 'Failed to delete subscriber'}), 500

@app.route('/api/send-sms', methods=['POST'])
def send_single_sms():
    try:
        phone = request.json.get('phone')
        message = request.json.get('message')
        
        if not phone or not message:
            return jsonify({'error': 'Phone number and message are required'}), 400
        
        # Check if Twilio is available
        if not TWILIO_AVAILABLE or not twilio_client:
            logger.warning("Twilio not available, SMS cannot be sent")
            return jsonify({
                'error': 'SMS service unavailable', 
                'message': 'SMS functionality is currently unavailable. Please try again later.'
            }), 503
            
        # Send SMS using Twilio
        twilio_client.messages.create(
            body=message,
            from_=os.getenv('TWILIO_PHONE'),
            to=phone
        )
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error sending SMS: {str(e)}")
        return jsonify({'error': 'Failed to send SMS'}), 500

@app.route('/api/send-bulk-sms', methods=['POST'])
def send_bulk_sms():
    try:
        message = request.json.get('message')
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Check if Twilio is available
        if not TWILIO_AVAILABLE or not twilio_client:
            logger.warning("Twilio not available, bulk SMS cannot be sent")
            return jsonify({
                'error': 'SMS service unavailable', 
                'message': 'SMS functionality is currently unavailable. Please try again later.'
            }), 503
        
        # Check if MongoDB is available to fetch subscribers
        if not db:
            logger.warning("MongoDB not available, cannot fetch subscribers for bulk SMS")
            return jsonify({
                'error': 'Database unavailable',
                'message': 'Cannot fetch subscribers. Please try again later.'
            }), 503
            
        subscribers = db.users.find({'phone': {'$exists': True}, 'subscribed': True})
        
        sent_count = 0
        error_count = 0
        
        for subscriber in subscribers:
            try:
                twilio_client.messages.create(
                    body=message,
                    from_=os.getenv('TWILIO_PHONE'),
                    to=subscriber['phone']
                )
                sent_count += 1
            except Exception as e:
                logger.error(f"Error sending SMS to {subscriber['phone']}: {str(e)}")
                error_count += 1
                continue
        
        return jsonify({
            'success': True,
            'sent': sent_count,
            'errors': error_count
        })
    except Exception as e:
        logger.error(f"Error in bulk SMS: {str(e)}")
        return jsonify({'error': 'Failed to send bulk SMS'}), 500

@app.route('/api/send-email', methods=['POST'])
def send_single_email():
    try:
        email = request.json.get('email')
        subject = request.json.get('subject')
        message = request.json.get('message')
        
        if not email or not subject or not message:
            return jsonify({'error': 'Email, subject, and message are required'}), 400
        
        # Check email configuration
        email_from = os.getenv('EMAIL_FROM')
        email_username = os.getenv('EMAIL_USERNAME')
        email_password = os.getenv('EMAIL_PASSWORD')
        smtp_server = os.getenv('SMTP_SERVER')
        smtp_port = os.getenv('SMTP_PORT')
        
        if not all([email_from, email_username, email_password, smtp_server, smtp_port]):
            logger.warning("Email configuration incomplete, email cannot be sent")
            return jsonify({
                'error': 'Email service unavailable', 
                'message': 'Email functionality is currently unavailable. Please try again later.'
            }), 503
            
        # Send email
        msg = MIMEMultipart()
        msg['From'] = email_from
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        
        try:
            server = smtplib.SMTP(smtp_server, int(smtp_port))
            server.starttls()
            server.login(email_username, email_password)
            server.send_message(msg)
            server.quit()
        except Exception as smtp_error:
            logger.error(f"SMTP error: {str(smtp_error)}")
            return jsonify({'error': 'Failed to send email due to SMTP error'}), 500
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return jsonify({'error': 'Failed to send email'}), 500

@app.route('/api/send-bulk-email', methods=['POST'])
def send_bulk_email():
    try:
        subject = request.json.get('subject')
        message = request.json.get('message')
        
        if not subject or not message:
            return jsonify({'error': 'Subject and message are required'}), 400
        
        # Check email configuration
        email_from = os.getenv('EMAIL_FROM')
        email_username = os.getenv('EMAIL_USERNAME')
        email_password = os.getenv('EMAIL_PASSWORD')
        smtp_server = os.getenv('SMTP_SERVER')
        smtp_port = os.getenv('SMTP_PORT')
        
        if not all([email_from, email_username, email_password, smtp_server, smtp_port]):
            logger.warning("Email configuration incomplete, email cannot be sent")
            return jsonify({
                'error': 'Email service unavailable', 
                'message': 'Email functionality is currently unavailable. Please try again later.'
            }), 503
        
        # Check if MongoDB is available to fetch subscribers
        if not db:
            logger.warning("MongoDB not available, cannot fetch subscribers for bulk email")
            return jsonify({
                'error': 'Database unavailable',
                'message': 'Cannot fetch subscribers. Please try again later.'
            }), 503
            
        subscribers = db.users.find({'email': {'$exists': True}, 'subscribed': True})
        subscriber_list = list(subscribers)  # Convert cursor to list to get count
        
        if len(subscriber_list) == 0:
            return jsonify({'message': 'No subscribers found'}), 200
        
        try:
            server = smtplib.SMTP(smtp_server, int(smtp_port))
            server.starttls()
            server.login(email_username, email_password)
            
            sent_count = 0
            error_count = 0
            
            for subscriber in subscriber_list:
                try:
                    msg = MIMEMultipart()
                    msg['From'] = email_from
                    msg['To'] = subscriber['email']
                    msg['Subject'] = subject
                    msg.attach(MIMEText(message, 'plain'))
                    server.send_message(msg)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"Error sending email to {subscriber['email']}: {str(e)}")
                    error_count += 1
                    continue
            
            server.quit()
            
            return jsonify({
                'success': True,
                'sent': sent_count,
                'errors': error_count,
                'total': len(subscriber_list)
            })
            
        except Exception as smtp_error:
            logger.error(f"SMTP error: {str(smtp_error)}")
            return jsonify({'error': 'Failed to send bulk email due to SMTP error'}), 500
            
    except Exception as e:
        logger.error(f"Error in bulk email: {str(e)}")
        return jsonify({'error': 'Failed to send bulk email'}), 500

@app.route('/api/subscribe', methods=['POST'])
def subscribe():
    try:
        data = request.json
        phone = data.get('phone')
        email = data.get('email')
        
        if not phone and not email:
            return jsonify({'error': 'Phone number or email is required'}), 400
        
        # Check if database is available
        if not db:
            logger.warning("MongoDB not available, cannot save subscription")
            return jsonify({
                'error': 'Database unavailable',
                'message': 'Subscription service is currently unavailable. Please try again later.',
                'db_status': 'unavailable'
            }), 503
            
        update_data = {'subscribed': True}
        if phone:
            update_data['phone'] = phone
        if email:
            update_data['email'] = email
            
        # Update or create user
        db.users.update_one(
            {'$or': [
                {'phone': phone} if phone else {'_id': None},
                {'email': email} if email else {'_id': None}
            ]},
            {'$set': update_data},
            upsert=True
        )
        
        # Send welcome messages - with error handling
        sms_status = None
        email_status = None
        
        # Send welcome SMS if phone provided
        if phone and TWILIO_AVAILABLE and twilio_client:
            try:
                twilio_client.messages.create(
                    body="Welcome to Krishimitra! You're now subscribed to our updates.",
                    from_=os.getenv('TWILIO_PHONE'),
                    to=phone
                )
                sms_status = 'sent'
            except Exception as e:
                logger.error(f"Error sending welcome SMS: {str(e)}")
                sms_status = 'failed'
        elif phone:
            sms_status = 'unavailable'
        
        # Send welcome email if email provided
        if email:
            # Check email configuration
            email_from = os.getenv('EMAIL_FROM')
            email_username = os.getenv('EMAIL_USERNAME')
            email_password = os.getenv('EMAIL_PASSWORD')
            smtp_server = os.getenv('SMTP_SERVER')
            smtp_port = os.getenv('SMTP_PORT')
            
            if all([email_from, email_username, email_password, smtp_server, smtp_port]):
                try:
                    msg = MIMEMultipart()
                    msg['From'] = email_from
                    msg['To'] = email
                    msg['Subject'] = 'Welcome to Krishimitra!'
                    msg.attach(MIMEText("Thank you for subscribing to Krishimitra! You'll now receive our updates.", 'plain'))
                    
                    server = smtplib.SMTP(smtp_server, int(smtp_port))
                    server.starttls()
                    server.login(email_username, email_password)
                    server.send_message(msg)
                    server.quit()
                    email_status = 'sent'
                except Exception as e:
                    logger.error(f"Error sending welcome email: {str(e)}")
                    email_status = 'failed'
            else:
                email_status = 'unavailable'
        
        return jsonify({
            'success': True,
            'sms_status': sms_status,
            'email_status': email_status
        })
    except Exception as e:
        logger.error(f"Error in subscribe: {str(e)}")
        return jsonify({'error': 'Failed to subscribe'}), 500

@app.route('/api/log-client-error', methods=['POST'])
def log_client_error():
    data = request.get_json()
    message = data.get('message', 'Unknown client error')
    source = data.get('source', 'client')
    stacktrace = data.get('stacktrace', '')
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    logger.error(f"Client error from {source}: {message}")
    if stacktrace:
        logger.error(f"Client stacktrace: {stacktrace}")
    
    return jsonify({'success': True})

# MongoDB connection
try:
    client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'), serverSelectionTimeoutMS=5000)
    # Verify connection
    client.server_info()
    db = client['krishimitra']
    logger.info("MongoDB connection successful")
except Exception as e:
    logger.error(f"MongoDB connection failed: {str(e)}")
    db = None

if __name__ == '__main__':
    logger.info("Starting KrishiMitra backend server")
    app.run(debug=True, host='0.0.0.0', port=5000) 