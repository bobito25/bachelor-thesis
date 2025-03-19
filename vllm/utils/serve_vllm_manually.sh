#!/bin/bash

# serve_vllm.sh

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

# Use HOST from environment if set; otherwise, use default or from PARAMS
HOST_FINAL="${HOST:-}"
if [ -n "$HOST_FINAL" ]; then
    # Remove existing --host from PARAMS if present
    NEW_PARAMS=()
    skip_next=false
    for ((i=0; i<${#PARAMS[@]}; i++)); do
        if $skip_next; then
            skip_next=false
            continue
        fi
        if [[ "${PARAMS[i]}" == "--host" ]]; then
            skip_next=true
            continue
        fi
        NEW_PARAMS+=("${PARAMS[i]}")
    done
    PARAMS=("${NEW_PARAMS[@]}")
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

# If HOST is provided via environment, add it to the command
if [ -n "$HOST_FINAL" ]; then
    COMMAND+=("--host" "$HOST_FINAL")
fi

# Add flags (parameters without values) from FLAGS array
if [ ${#FLAGS[@]} -gt 0 ]; then
    for flag in "${FLAGS[@]}"; do
        if [ -n "$flag" ]; then
            COMMAND+=("$flag")
        fi
    done
fi
echo "Conda environment:"
echo $CONDA_DEFAULT_ENV
#also output conda env list
echo "Conda env list:"
conda env list
bash $UTILS/activate_create_env.sh &&
echo "Conda environment activate successfully" 
export VLLM_USE_V1=1
# Log the constructed command
log "Executing command: ${COMMAND[*]}"

# Execute the command
"${COMMAND[@]}"
