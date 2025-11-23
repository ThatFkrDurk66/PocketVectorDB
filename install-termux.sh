#!/data/data/com.termux/files/usr/bin/bash

# Termux Installation Script for PocketVectorDB
# This script installs the vector database and its dependencies in Termux

set -e

echo "=================================="
echo "PocketVectorDB Termux Installer"
echo "=================================="
echo ""

# Check if running in Termux
if [ ! -d "/data/data/com.termux" ]; then
    echo "Warning: This script is designed for Termux environment."
    echo "Continuing anyway..."
fi

# Update package list
echo "Updating package list..."
pkg update -y || apt update -y

# Install Python if not installed
echo "Checking Python installation..."
if ! command -v python &> /dev/null; then
    echo "Installing Python..."
    pkg install -y python || apt install -y python
fi

# Install build dependencies
echo "Installing build dependencies..."
pkg install -y python-pip || apt install -y python-pip
pkg install -y binutils || apt install -y binutils
pkg install -y cmake || apt install -y cmake

# Install numpy dependencies (OpenBLAS for faster numpy operations)
echo "Installing numpy dependencies..."
pkg install -y openblas || apt install -y openblas

# Install numpy first (compiled version for Termux)
echo "Installing numpy..."
pip install numpy>=1.21.0

# Install the vector database
echo "Installing PocketVectorDB..."
pip install -e .

echo ""
echo "=================================="
echo "Installation Complete!"
echo "=================================="
echo ""
echo "To test the installation, run:"
echo "  python example_usage.py"
echo ""
echo "Or use in Python:"
echo "  from vectordb import VectorDB"
echo "  db = VectorDB('./my_vectors')"
echo ""
echo "For documentation, see README.md"
echo ""
