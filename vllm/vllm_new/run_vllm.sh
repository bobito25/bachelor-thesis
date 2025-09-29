#!/bin/bash

cd /work/rleap1/leon.hamm/vllm_new

# Load secrets
set -a  # automatically export all variables
source .env
set +a

echo "Base path: $BASE_PATH"

# Set custom temporary directory
export TMPDIR="$BASE_PATH/tmp"
export TEMP="$BASE_PATH/tmp"
export TMP="$BASE_PATH/tmp"

# Create necessary directories
mkdir -p "$TMPDIR"

UV_BIN_DIR="$BASE_PATH/uv_bin"
export UV_CACHE_DIR="$BASE_PATH/uv_cache_dir"

# Check if 'uv' is already available in PATH or in the target dir
if command -v uv &> /dev/null || [ -x "$UV_BIN_DIR/uv" ]; then
    echo "'uv' is already installed."
else
    echo "'uv' is not installed. Installing to $UV_BIN_DIR..."

    # Ensure curl is available
    if ! command -v curl &> /dev/null; then
        echo "Error: curl is not installed. Please install curl first."
        exit 1
    fi

    # Run the official install script with the custom install path
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_BIN_DIR" sh
fi

# Add uv to PATH
if [[ ":$PATH:" != *":$UV_BIN_DIR:"* ]]; then
    export PATH="$UV_BIN_DIR:$PATH"
    echo "Added $UV_BIN_DIR to PATH."
else
    echo "$UV_BIN_DIR is already in PATH."
fi

# Path to the virtual environment
VENV_DIR=".venv"

# Check if the virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR."
else
    echo "Creating virtual environment with Python 3.12 using uv..."
    uv venv --python 3.12 --seed
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Make sure Python 3.12 is installed and uv is available."
        exit 1
    fi
fi

# Activate the virtual environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export VLLM_USE_TRITON_FLASH_ATTN=0

# Install vllm with appropriate torch backend
echo "Installing vllm"
uv pip install --pre vllm==0.10.1+gptoss \
    --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --index-strategy unsafe-best-match

export HF_HOME=$BASE_PATH/huggingface_cache

mkdir -p "$HF_HOME"

# Exit immediately if a command exits with a non-zero status
set -e

# Function to log messages with timestamp
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Ensure CONFIG_FILE is set in the environment
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: CONFIG_FILE environment variable is not set."
    exit 1
fi

# Check if the configuration file exists and is readable
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

if [ ! -r "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' is not readable."
    exit 1
fi

# Load the configuration
set -a  # Automatically export all variables
source "$CONFIG_FILE"
set +a

# Verify mandatory variable
if [ -z "$MODEL_NAME" ]; then
    echo "Error: 'MODEL_NAME' is not set in '$CONFIG_FILE'."
    exit 1
fi

# Initialize the command with 'vllm serve' and the MODEL_NAME
COMMAND=("vllm" "serve" "$MODEL_NAME")

# Add parameters with values from PARAMS array
if [ ${#PARAMS[@]} -gt 0 ]; then
    for ((i=0; i<${#PARAMS[@]}; i+=2)); do
        flag="${PARAMS[i]}"
        value="${PARAMS[i+1]}"
        if [ -n "$flag" ] && [ -n "$value" ]; then
            COMMAND+=("$flag" "$value")
        else
            echo "Warning: Incomplete parameter pair at position $i in PARAMS array."
        fi
    done
fi

# Add flags (parameters without values) from FLAGS array
if [ ${#FLAGS[@]} -gt 0 ]; then
    for flag in "${FLAGS[@]}"; do
        if [ -n "$flag" ]; then
            COMMAND+=("$flag")
        fi
    done
fi

echo "Current ulimit -n:"
ulimit -n
# Set ulimit to a higher value if necessary
echo "Setting ulimit -n to 65536"
ulimit -n 65536

log "Executing command: ${COMMAND[*]}"

# Execute the command
"${COMMAND[@]}"
