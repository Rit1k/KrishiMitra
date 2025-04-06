#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies
apt-get update
apt-get install -y build-essential python3.11-dev python3.11-venv wget

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install Python packages
pip install --no-cache-dir -r requirements.txt

# Create backend directory if it doesn't exist
mkdir -p backend

# Download the model file
echo "Downloading plant disease model..."
wget -O backend/plant_disease_model.h5 https://storage.googleapis.com/krishimitra-models/plant_disease_model.h5

# Verify the model file was downloaded
if [ -f "backend/plant_disease_model.h5" ]; then
    echo "Model file downloaded successfully"
else
    echo "Error: Failed to download model file"
    exit 1
fi 