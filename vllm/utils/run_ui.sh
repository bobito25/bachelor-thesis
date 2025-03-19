#!/bin/bash

# run_ui.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Ensure PORT_UI is set in the environment
if [ -z "$PORT_UI" ]; then
    echo "Error: PORT_UI environment variable is not set."
    exit 1
fi

# Check if config.conf exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found."
    exit 1
fi

# Source the config.conf
source "$CONFIG_FILE"

# Function to extract parameter value from PARAMS array
get_param() {
    local param="$1"
    local array=("${PARAMS[@]}")
    for ((i=0; i < ${#array[@]}; i+=2)); do
        if [[ "${array[i]}" == "$param" ]]; then
            echo "${array[i+1]}"
            return
        fi
    done
}

# Extract HOST and PORT from PARAMS
HOST=$(get_param "--host")
PORT=$(get_param "--port")

# Validate HOST and PORT
if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "Error: HOST or PORT not set in PARAMS."
    exit 1
fi

# Check if MODEL_NAME is set
if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME is not set in config.conf."
    exit 1
fi

# Check if NGROK_AUTH_TOKEN is set in environment
if [ -z "$NGROK_AUTH_TOKEN" ]; then
    echo "Error: NGROK_AUTH_TOKEN environment variable is not set."
    exit 1
fi

# Construct model-url
MODEL_URL="http://localhost:${PORT}/v1"

# Assemble PARAMS excluding --host and --port
EXTRA_PARAMS=()
for ((i=0; i < ${#PARAMS[@]}; i+=2)); do
    key="${PARAMS[i]}"
    value="${PARAMS[i+1]}"
    if [[ "$key" != "--host" && "$key" != "--port" ]]; then
        EXTRA_PARAMS+=("$key" "$value")
    fi
done

# Assemble FLAGS
EXTRA_FLAGS=("${FLAGS[@]}")

pip install gradio
#pip install pddl
pip install ngrok
pip install openai

# Execute the Python script with all parameters and flags
python -u $UTILS/gradio_interface_image.py \
    --host "$HOST" \
    --port "$PORT_UI" \
    --model-url "$MODEL_URL" \
    --model "$MODEL_NAME" \
    --ngrok_token "$NGROK_AUTH_TOKEN" \
    --ngrok_domain_url "$NGROK_WEBSITE" \
