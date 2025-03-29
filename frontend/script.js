// API endpoint configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const phoneInput = document.getElementById('phone-number');
const subscribeBtn = document.querySelector('.subscribe-btn');
const getStartedBtns = document.querySelectorAll('.get-started');

// Feature card click handlers
getStartedBtns.forEach((btn, index) => {
    btn.addEventListener('click', () => {
        const features = ['disease-detection', 'weather', 'transport', 'chatbot'];
        window.location.href = `/${features[index]}.html`;
    });
});

// SMS Subscription handler
subscribeBtn.addEventListener('click', async () => {
    const phone = phoneInput.value;
    if (!phone) {
        alert('Please enter a phone number');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/subscribe`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ phone }),
        });

        const data = await response.json();
        if (response.ok) {
            alert('Successfully subscribed to SMS updates!');
            phoneInput.value = '';
        } else {
            alert(data.error || 'Failed to subscribe. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again later.');
    }
});

// Weather monitoring
async function getWeather(lat, lon) {
    try {
        const response = await fetch(`${API_BASE_URL}/weather?lat=${lat}&lon=${lon}`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching weather:', error);
        return null;
    }
}

// Transport services
async function getNearbyTransporters(lat, lon) {
    try {
        const response = await fetch(`${API_BASE_URL}/transporters?lat=${lat}&lon=${lon}`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching transporters:', error);
        return null;
    }
}

// Disease detection
async function detectDisease(imageFile) {
    try {
        const formData = new FormData();
        formData.append('image', imageFile);

        const response = await fetch(`${API_BASE_URL}/detect-disease`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error detecting disease:', error);
        return null;
    }
}

// Chatbot functionality
async function sendMessage(message, userId) {
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message,
                user_id: userId,
            }),
        });

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error sending message:', error);
        return null;
    }
}

// Geolocation
function getCurrentLocation() {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
            reject(new Error('Geolocation is not supported by your browser'));
            return;
        }

        navigator.geolocation.getCurrentPosition(
            (position) => {
                resolve({
                    lat: position.coords.latitude,
                    lon: position.coords.longitude,
                });
            },
            (error) => {
                reject(error);
            }
        );
    });
}

// Initialize the application
async function initApp() {
    try {
        // Get user's location for weather and transport services
        const location = await getCurrentLocation();
        if (location) {
            // Initialize weather and transport services with location
            const weather = await getWeather(location.lat, location.lon);
            const transporters = await getNearbyTransporters(location.lat, location.lon);
            
            // Update UI with weather and transport information
            // (Implementation depends on specific UI components)
        }
    } catch (error) {
        console.error('Error initializing app:', error);
    }
}

// Call initApp when the page loads
document.addEventListener('DOMContentLoaded', initApp); 