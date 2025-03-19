#!/bin/bash

# Function to extract the environment name from environment.yml
extract_env_name() {
    # Attempt to extract the 'name' field from the YAML file
    ENV_NAME=$(grep -E '^name:' "$ENV_YML_PATH" | awk '{print $2}')

    # Check if ENV_NAME was successfully extracted
    if [ -z "$ENV_NAME" ]; then
        echo "Error: Could not find 'name' field in '$ENV_YML_PATH'. Please ensure the environment.yml contains a 'name' field."
        exit 1
    fi
}

echo "Checking if conda environment: $ENV_NAME exists and activating it"
# Function to check if a Conda environment exists
env_exists() {
    conda env list | awk '{print $1}' | grep -w "$ENV_NAME" > /dev/null 2>&1
}


# Function to create the Conda environment from environment.yml
create_env() {
    echo "Creating Conda environment '$ENV_NAME' from '$ENV_YML_PATH'..."
    conda env create -f "$ENV_YML_PATH"
    if [ $? -eq 0 ]; then
        echo "Conda environment '$ENV_NAME' created successfully."
    else
        echo "Error: Failed to create Conda environment '$ENV_NAME'."
        exit 1
    fi
}

# Function to activate the Conda environment
activate_env() {
    echo "Activating Conda environment '$ENV_NAME'..."
    # Initialize Conda for the current shell session
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    # pip uninstall transformers -y
    # pip install 'git+https://github.com/huggingface/transformers.git'
    # pip install --upgrade vllm
    if [ $? -eq 0 ]; then
        echo "Conda environment '$ENV_NAME' is now active."
    else
        echo "Error: Failed to activate Conda environment '$ENV_NAME'."
        exit 1
    fi
}

# Check if the environment.yml file exists
if [ ! -f "$ENV_YML_PATH" ]; then
    echo "Error: '$ENV_YML_PATH' does not exist. Please ensure the environment.yml file is present."
    exit 1
fi

# Extract the environment name from the environment.yml file
extract_env_name

# Check if the environment exists
if env_exists; then
    echo "Conda environment '$ENV_NAME' already exists."
    activate_env
else
    create_env
    activate_env
fi

# Optional: Update the environment if environment.yml has changed
# Uncomment the following lines if you want to always update the environment
# echo "Updating Conda environment '$ENV_NAME' with '$ENV_YML_PATH'..."
# conda env update -f "$ENV_YML_PATH" --prune
# echo "Conda environment '$ENV_NAME' updated successfully."

# Final message
echo "Conda environment '$ENV_NAME' is now active and ready to use."
