#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120g
#SBATCH --gres=gpu:4
#SBATCH --account=rleap
#SBATCH --partition=rleap_gpu_24gb
#SBATCH --time=4:00:00
#SBATCH --output=output_files/job-%A.out
#SBATCH --error=output_files/job-%A.err

which nvcc
nvcc --version
nvidia-smi

export CUDA_HOME=/usr/local/cuda-12.1

bash run_vllm.sh
