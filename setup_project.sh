#!/bin/bash

# Define the virtual environment directory
VENV_DIR=".venv"

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install Python3 to proceed."
    exit 1
fi

# Create the virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment already exists."
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found!"
    deactivate
    exit 1
fi

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Check for successful installation
if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully."
else
    echo "Failed to install dependencies."
fi

echo "Downloading and preparing additional supporting dependencies..."
echo "python -m spacy download en_core_web_md"
python -m spacy download en_core_web_md
echo "./scripts/gather_msmarco_passage_data.sh"
./scripts/gather_msmarco_passage_data.sh

# Deactivate virtual environment
deactivate

echo "Project setup complete."
