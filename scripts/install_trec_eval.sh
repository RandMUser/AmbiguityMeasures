#!/bin/bash

# Configurable installation path for trec_eval
TREC_EVAL_INSTALL_PATH="./.venv/bin"

# Define saved variables
RELATIVE_PATH="$TREC_EVAL_INSTALL_PATH"  # Replace with actual relative path from the saved working directory

# Function to save the current working directory to the SAVED_WORKING_DIRECTORY variable
save_current_working_directory() {
  SAVED_WORKING_DIRECTORY="$(pwd)"
  echo "Saved current working directory: $SAVED_WORKING_DIRECTORY"
}

# Function to copy a file to a target directory
copy_file_to_target() {
  local file_to_copy="$1"
  local target_directory="$2"

  cp "$file_to_copy" "$target_directory"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to copy $file_to_copy to $target_directory."
    exit 1
  fi

  echo "Successfully copied $file_to_copy to $target_directory."
}

# Function to install trec_eval if not available
install_trec_eval() {
  echo "trec_eval not found. Cloning and building trec_eval..."
  TEMP_DIR=$(mktemp -d)
  git clone https://github.com/usnistgov/trec_eval.git "$TEMP_DIR"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to clone trec_eval repository."
    exit 1
  fi

  cd "$TEMP_DIR" || exit
  make
  if [ $? -ne 0 ]; then
    echo "Error: Failed to build trec_eval."
    exit 1
  fi

  # Create the full target directory path
  TARGET_DIRECTORY="$SAVED_WORKING_DIRECTORY/$RELATIVE_PATH"
  # Copy file from current working directory to target directory
  FILE_TO_COPY="trec_eval"  # Replace 'filename' with the actual file you want to copy
  copy_file_to_target "$FILE_TO_COPY" "$TARGET_DIRECTORY"
}

# Main script execution
save_current_working_directory

# Check if trec_eval exists in PATH or specified install path
if ! command -v trec_eval &> /dev/null && [ ! -f "$TREC_EVAL_INSTALL_PATH/trec_eval" ]; then
  install_trec_eval
fi

# Use trec_eval from the install path if not found in PATH
if ! command -v trec_eval &> /dev/null; then
  TREC_EVAL_COMMAND="$TREC_EVAL_INSTALL_PATH/trec_eval"
else
  TREC_EVAL_COMMAND="trec_eval"
fi

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
  echo "Usage: ./run_trec_eval.sh <qrel_filepath> <run_filepath>"
  exit 1
fi

QREL_FILE=$1
RUN_FILE=$2

# Execute trec_eval
$TREC_EVAL_COMMAND "$QREL_FILE" "$RUN_FILE"
