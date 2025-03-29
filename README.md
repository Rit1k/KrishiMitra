# Farmer's Digital Assistant

A comprehensive digital platform designed to empower farmers with modern technology and AI-driven insights. This application provides various features to help farmers make informed decisions about their agricultural practices.

## Features

1. **Crop Disease Detection**
   - Use phone camera to detect crop diseases
   - Get instant AI-powered treatment recommendations
   - Supports multiple crop types

2. **Weather Monitoring**
   - Real-time weather updates
   - Soil analysis integration
   - Weather forecasting for farming decisions

3. **Transport Services**
   - Find nearby transporters
   - Logistics services for agricultural produce
   - Google Maps integration

4. **AI Assistant (Chatbot)**
   - Personalized farming advice
   - Powered by Google's Gemini AI
   - Multi-language support
   - Limited free queries with subscription option

5. **SMS Subscription Service**
   - Daily updates about weather
   - Crop disease alerts
   - Market price information
   - Personalized farming tips

## Technology Stack

- **Backend**: Python (Flask)
- **Frontend**: HTML, CSS, JavaScript
- **Database**: MongoDB
- **APIs**:
  - OpenAI API
  - Google Gemini AI
  - OpenWeather API
  - Google Maps API
  - Twilio SMS API

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/farmers-digital-assistant.git
   cd farmers-digital-assistant
   ```

2. **Set up Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file and add your API keys and configurations.

4. **Set up MongoDB**
   - Install MongoDB if not already installed
   - Start MongoDB service
   - The application will automatically create required collections

5. **Start the backend server**
   ```bash
   flask run
   ```

6. **Open the frontend**
   - Open `frontend/index.html` in your web browser
   - Or serve it using a local server:
     ```bash
     python -m http.server 3000
     ```

## API Keys Required

1. **OpenAI API Key** - For text processing
2. **Google Gemini API Key** - For AI assistant
3. **OpenWeather API Key** - For weather data
4. **Google Maps API Key** - For transport services
5. **Twilio Account** - For SMS services
   - Account SID
   - Auth Token
   - Phone Number

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, email support@farmersdigitalassistant.com or join our Telegram group. 