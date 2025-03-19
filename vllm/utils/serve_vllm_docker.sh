#!/bin/bash

# Check if Apptainer/Singularity is available
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
    echo "Apptainer found."
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
    echo "Singularity found."
else
    echo "Neither Apptainer nor Singularity found. Please install one of them."
    exit 1
fi

#Pull if not already pulled
$CONTAINER_CMD pull --dir $BASE_PATH/vllm_project docker://vllm/$VLLM_VERSION



# start_vllm_server.sh

# Exit immediately if a command exits with a non-zero status
set -e

# Load configuration
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Configuration file $CONFIG_FILE not found!"
  exit 1
fi
source "$CONFIG_FILE"

# Construct PARAMS string
PARAMS_STR=""
for ((i=0; i<${#PARAMS[@]}; i+=2)); do
  PARAMS_STR+=" ${PARAMS[i]} ${PARAMS[i+1]}"
done

# Construct FLAGS string
FLAGS_STR=""
for flag in "${FLAGS[@]}"; do
  FLAGS_STR+=" $flag"
done


# Start the vLLM server

echo "Starting vLLM server..."

# Set the command to use Apptainer instead of Docker
CONTAINER_CMD="apptainer"

# GPU support requires the --nv flag
GPU_FLAG="--nv"

#replace : with _ in version
SANITIZED_VLLM_VERSION="${VLLM_VERSION//:/_}"
IMAGE_PATH="$BASE_PATH/vllm_project/${SANITIZED_VLLM_VERSION}.sif"
echo "IMAGE_PATH: $IMAGE_PATH"

# Start the vLLM server
echo "Starting vLLM server..."

# GPU support requires the --nv flag
GPU_FLAG="--nv"

# Run the Apptainer container with proper flags and command
apptainer run $GPU_FLAG \
    --env HF_HOME="$HF_HOME" \
    --env HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    --bind "$PWD:/workspace" \
    --bind "/work:/work" \
    --workdir /workspace \
    "$IMAGE_PATH" \
    --model "$MODEL_NAME" \
    $PARAMS_STR \
    $FLAGS_STR &




# Capture the PID of the background process
echo "vLLM server started with PID $!"

