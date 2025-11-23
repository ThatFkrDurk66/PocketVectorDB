#!/data/data/com.termux/files/usr/bin/bash

# Install from pre-built wheel (fastest method)
# Copy the .whl file to your Termux device first

echo "Installing Personal VectorDB from wheel..."

# Find the wheel file
WHEEL_FILE=$(ls dist/personal_vectordb-*.whl 2>/dev/null | head -n 1)

if [ -z "$WHEEL_FILE" ]; then
    echo "Error: Wheel file not found in dist/ directory"
    echo "Please ensure the .whl file is present"
    exit 1
fi

# Install dependencies first
echo "Installing dependencies..."
pip install numpy

# Install the package
echo "Installing from $WHEEL_FILE..."
pip install "$WHEEL_FILE"

echo ""
echo "Installation complete!"
echo "Test with: python -c 'from vectordb import VectorDB; print(VectorDB)'"
