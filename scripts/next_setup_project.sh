#!/bin/bash
# an update to setup_project.sh that incorporates trec_eval check and install as a separate setup script versus apt-get target.
# TODO: needs to be tested and moved into the root dir to replace setup_project.sh

# Define the virtual environment directory
VENV_DIR=".venv"

# Dictionary of installation commands (command -> package name or install script)
declare -A INSTALL_COMMANDS=(
    ["python3"]="python3"
    ["curl"]="curl"
    ["trec_eval"]="./scripts/install_trec_eval.sh"
)

# Run apt-get update once
sudo apt-get update

# Function to check for prerequisite commands
check_command() {
    if ! command -v "$1" &> /dev/null
    then
        echo "$1 is not installed. Attempting to install $1..."
        install_command "$1"
    fi
}

# Function to install prerequisite commands
install_command() {
    if [[ -n "${INSTALL_COMMANDS[$1]}" ]]; then
        if [[ "${INSTALL_COMMANDS[$1]}" == *".sh" ]]; then
            # If the value is a script, run the script to install the command
            bash "${INSTALL_COMMANDS[$1]}"
        else
            # Otherwise, use apt-get to install the package
            sudo apt-get install -y "${INSTALL_COMMANDS[$1]}"
        fi
    else
        echo "No installation command defined for $1. Please install it manually."
        exit 1
    fi
}

# List of prerequisite commands
PREREQUISITES=("python3" "curl" "trec_eval")

# Check for all prerequisites
for cmd in "${PREREQUISITES[@]}"; do
    check_command "$cmd"
done

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
    deactivate
    exit 1
fi

# Download and prepare additional supporting dependencies
echo "Downloading and preparing additional supporting dependencies..."
echo "python -m spacy download en_core_web_md"
python -m spacy download en_core_web_md
echo "./scripts/gather_msmarco_passage_data.sh"
./scripts/gather_msmarco_passage_data.sh

# Deactivate virtual environment
deactivate

echo "Project setup complete."
