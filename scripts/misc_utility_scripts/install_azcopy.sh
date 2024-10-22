#!/bin/bash

# Check if azcopy is installed
if ! command -v azcopy &> /dev/null; then
    echo "AzCopy is not installed. Installing AzCopy..."

    # Download the latest AzCopy
    wget -q https://aka.ms/downloadazcopy-v10-linux -O azcopy_linux.tar.gz

    # Extract the tar.gz file
    tar -xf azcopy_linux.tar.gz

    # Move the AzCopy binary to /usr/local/bin
    sudo mv azcopy_linux_amd64_*/azcopy /usr/local/bin/

    # Set executable permissions
    sudo chmod +x /usr/local/bin/azcopy

    # Verify installation
    if command -v azcopy &> /dev/null; then
        echo "AzCopy installed successfully."
    else
        echo "Failed to install AzCopy."
        exit 1
    fi

    # Clean up
    rm -rf azcopy_linux_amd64_* azcopy_linux.tar.gz
else
    echo "AzCopy is already installed."
fi
