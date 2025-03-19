#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e


# Create the target directory if it doesn't exist
mkdir -p "$BASE_PATH"
echo "Using target directory: $BASE_PATH"

# Navigate to the target directory
cd "$BASE_PATH"

# Function to check if a Python package is installed
is_installed() {
    pip show "$1" &> /dev/null
}

# Clone the vllm repository if it doesn't exist
if [ ! -d "vllm" ]; then
    echo "Cloning vllm repository..."
    git clone https://github.com/vllm-project/vllm.git
else
    echo "vllm repository already exists. Skipping clone."
fi

# Navigate into the vllm directory
cd vllm

# Install vllm in editable mode if not already installed
if is_installed "vllm"; then
    echo "vllm is already installed. Skipping editable install."
else
    pip install vllm
fi

# Navigate back to the target directory
cd ..

# Install flash-attn if not already installed
# if is_installed "flash-attn"; then
#     echo "flash-attn is already installed. Skipping install."
# else
#     echo "Installing flash-attn..."
#     pip install flash-attn
# fi

# Install vllm via pip if not already installed (redundant if installed in editable mode)
# This step can be skipped if installing in editable mode is sufficient
if is_installed "vllm"; then
    echo "vllm is already installed via pip. Skipping install."
else
    echo "Installing vllm via pip..."
    pip install vllm
    
fi
#install to be sure tough but upgradae
pip install --upgrade transformers vllm 
echo "Setup completed successfully."
