#!/bin/bash

# Load secrets
set -a  # automatically export all variables
source .env
set +a

# Get user name
USER_NAME=$(whoami)
export USER_NAME

# Set BASE_PATH: Use value from .env if defined, else use default path and export
if [ -z "$BASE_PATH" ]; then
  BASE_PATH="/work/rleap1/$USER_NAME"
  export BASE_PATH
fi

# Set cache directories
export APPTAINER_CACHEDIR=$BASE_PATH/apptainer_cache
export GENERAL_CACHE_DIR=$BASE_PATH/cache
export PIP_CACHE_DIR=$BASE_PATH/cache/pip
export HF_HOME=$BASE_PATH/huggingface_cache

# Set temporary build directory
export SINGULARITY_TMPDIR=$BASE_PATH/singularity_tmp

# Create directories if they do not exist
mkdir -p "$APPTAINER_CACHEDIR"
mkdir -p "$GENERAL_CACHE_DIR"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$HF_HOME"
mkdir -p "$SINGULARITY_TMPDIR"

#get user name
USER_NAME=$(whoami)
export USER_NAME


# Set custom temporary directory
export TMPDIR="$BASE_PATH/tmp"
export TEMP="$BASE_PATH/tmp"
export TMP="$BASE_PATH/tmp"

# Set pip cache and build directories
export PIP_CACHE_DIR="$BASE_PATH/pip_cache"
export PIP_BUILD="$BASE_PATH/pip_build"

# Create necessary directories
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$PIP_BUILD"

# Configure pip to use the custom cache directory
# Create or overwrite pip.conf to set cache-dir globally
mkdir -p "$BASE_PATH/.pip"
cat > "$BASE_PATH/.pip/pip.conf" <<EOL
[global]
cache-dir = $PIP_CACHE_DIR
build = $PIP_BUILD
EOL

# Export PIP_CONFIG_FILE to ensure pip uses the custom configuration
export PIP_CONFIG_FILE="$BASE_PATH/.pip/pip.conf"

# Optionally, ensure pip uses the custom config by default
export PIP_NO_CACHE_DIR=false

#create folder if not already created
# Call the setup_cache.sh script
bash $UTILS/create_init_folders.sh && 
echo "Folders created successfully" 


# Download and install Miniconda
MINICONDA_PATH="$BASE_PATH/miniconda3"
export MINICONDA_PATH

echo installing miniconda in $MINICONDA_PATH
bash $UTILS/create_miniconda_path.sh && 
echo "Miniconda installed successfully"

# Add Miniconda to PATH and initialize
export PATH="$MINICONDA_PATH/bin:$PATH"
source "$MINICONDA_PATH/etc/profile.d/conda.sh"

# Remove existing envs_dirs settings
conda config --remove-key envs_dirs

# Add the desired envs_dirs path
conda config --add envs_dirs $MINICONDA_PATH/envs

#path to environment.yml
ENV_YML_PATH="$UTILS/environment.yml"
export ENV_YML_PATH
echo "Installing conda environment from $ENV_YML_PATH"

#create and activate conda environment
bash $UTILS/activate_create_env.sh &&
echo "Conda environment created successfully" 
echo starting the UI if specified

if [ "$START_UI" = "true" ]; then
  #start the UI
  echo starting the UI
  bash $UTILS/run_ui.sh &
fi 
#
pip install "bitsandbytes>=0.44.0"
# export VLLM_LOGGING_LEVEL=DEBUG
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=TRACE
# export VLLM_TRACE_FUNCTION=1
echo uninstalling flash attantion
pip uninstall flash-attn -y
#echo installing transformers
#pip install --no-cache-dir transformers==4.48.1

echo starting the server

#TODO: do in wiki vlm hosting it stuff
if [ "$USE_DOCKER" = "true" ]; then
  echo "Using Docker"
  #install if not already installed
  bash $UTILS/install_vllm_docker.sh 
  #start the server
  bash $UTILS/serve_vllm_docker.sh 
else
  echo "Using Python"
  #install if not already installed
  bash $UTILS/install_vllm_python.sh 
  #start the server
  bash $UTILS/serve_vllm_manually.sh
fi
echo start the server successfully
echo guess we are wating now
wait
