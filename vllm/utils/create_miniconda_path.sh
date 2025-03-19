#!/bin/bash

#install miniconda if not already installed and create folder
if [ ! -d "$MINICONDA_PATH" ]; then
    echo "Downloading and installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $BASE_PATH/miniconda.sh
    bash $BASE_PATH/miniconda.sh -b -p $MINICONDA_PATH
    rm $BASE_PATH/miniconda.sh
fi