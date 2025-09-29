# interactive chat with vllm client

echo "Starting job on $(hostname) at $(date)"

# activate python env
source /work/rleap1/leon.hamm/bachelor-thesis-dev/novelty_tot/.venv/bin/activate

# Find all running SLURM jobs with name "run_vllm_cluster2.slurm" and get their nodes
VLLM_NODES=$(squeue -h -o "%N" -n "run_vllm_cluster2.slurm" -t RUNNING | tr '\n' ',' | sed 's/,$//')

if [ -n "$VLLM_NODES" ]; then
    echo "Found VLLM jobs running on nodes: $VLLM_NODES"
    export VLLM_NODES
else
    echo "No running VLLM jobs found. Exiting."
    exit 1
fi

# run job
python -u vllm_chat.py