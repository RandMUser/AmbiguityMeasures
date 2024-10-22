#!/bin/bash

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

# Define saved variables
save_current_working_directory  # Save the current working directory
RELATIVE_PATH=".venv/bin"  # Replace with actual relative path from the saved working directory

# Create the full target directory path
TARGET_DIRECTORY="$SAVED_WORKING_DIRECTORY/$RELATIVE_PATH"

# Ensure the target directory exists
if [ ! -d "$TARGET_DIRECTORY" ]; then
  echo "Creating target directory: $TARGET_DIRECTORY"
  mkdir -p "$TARGET_DIRECTORY"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create target directory: $TARGET_DIRECTORY"
    exit 1
  fi
fi

# Copy file from current working directory to target directory
FILE_TO_COPY="trec_eval"  # Replace 'filename' with the actual file you want to copy
copy_file_to_target "$FILE_TO_COPY" "$TARGET_DIRECTORY"