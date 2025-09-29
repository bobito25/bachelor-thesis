#!/bin/bash
#SBATCH --job-name=novtot_tests     # Job name
#SBATCH --output=output_files/output_%j.log      # Output log file (%j = job ID)
#SBATCH --error=output_files/error_%j.log        # Error log file
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --cpus-per-task=12          # Number of CPUs per task
#SBATCH --mem=8G                    # Memory per node
#SBATCH --time=12:00:00             # Time limit (hh:mm:ss)
#SBATCH --partition=rleap_cpu       # Partition/queue name

echo "Starting job on $(hostname) at $(date)"

# activate python env
source /work/rleap1/leon.hamm/bachelor-thesis-dev/novelty_tot/.venv/bin/activate

# Find all running SLURM jobs with name "run_vllm_cluster.slurm" and get their nodes
VLLM_NODES=$(squeue -h -o "%N" -n "run_vllm_cluster2.slurm" -t RUNNING | tr '\n' ',' | sed 's/,$//')

if [ -n "$VLLM_NODES" ]; then
    echo "Found VLLM jobs running on nodes: $VLLM_NODES"
    export VLLM_NODES
else
    echo "No running VLLM jobs found. Exiting."
    exit 1
fi

# run job
python -u run_batch_jobs.py
