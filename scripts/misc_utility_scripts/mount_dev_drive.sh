#!/bin/bash
# Script setup to replicate the virtual development target architecture used on the CCR Laptop (489 Lab) with Virtual Box Ubuntu Desktop Guest with a separate .vhd for development code/data/experiment artifacts...
# This setup is intended to help separate the guest OS filesystem from code/data/artifacts needed for an experiment which will help with data backup, portability, etc. use cases.
# TODO: Needs to be tested...
# Check if the user has provided the correct number of arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <drive> <mount_path> <link_path>"
    exit 1
fi

# Arguments
drive=$1
mount_path=$2
link_path=$3

# Get UUID of the drive
uuid=$(blkid -s UUID -o value $drive)
if [ -z "$uuid" ]; then
    echo "Error: Could not determine UUID for $drive"
    exit 1
fi

# Create the mount path if it does not exist
if [ ! -d "$mount_path" ]; then
    echo "Creating mount path at $mount_path"
    mkdir -p "$mount_path"
fi

# Append entry to /etc/fstab
fstab_entry="UUID=$uuid $mount_path ext4 defaults 0 2"
echo "Adding entry to /etc/fstab: $fstab_entry"
echo "$fstab_entry" | sudo tee -a /etc/fstab > /dev/null

# Mount the filesystem
sudo mount -a
if [ $? -ne 0 ]; then
    echo "Error: Failed to mount filesystem"
    exit 1
fi

# Create the symbolic link
if [ -e "$link_path" ]; then
    echo "Warning: $link_path already exists. Removing it."
    sudo rm -rf "$link_path"
fi

echo "Creating symbolic link from $mount_path to $link_path"
sudo ln -s "$mount_path" "$link_path"

# Final confirmation
echo "Drive $drive successfully mounted at $mount_path and linked to $link_path"
exit 0
