run_name="test"
log_dir="/data/logs/pmllm"

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

timestamp=$(date +"%Y.%m.%d_%H.%M.%S")
run_name="${timestamp}_${run_name}"
log_dir="${log_dir}/${run_name}"

config_path="scripts/configs/train.yaml"

source .venv/bin/activate

# Check if resume_from_checkpoint is true in the config file
resume_from_checkpoint=$(grep "resume_from_checkpoint:" ${config_path} | grep -o "true\|false")

if [[ "$resume_from_checkpoint" == "true" ]]; then
    # If resuming, get log_dir from checkpoint path by removing the last two path components
    # Match only the model-level ckpt_path (with 2-space indentation), not the one in encoder_kwargs
    ckpt_path=$(grep "^  ckpt_path:" ${config_path} | awk '{print $2}')
    if [[ -n "$ckpt_path" ]]; then
        # Remove filename and checkpoints directory to get the log directory
        log_dir=$(dirname "$(dirname "$ckpt_path")")
    fi
    
    # Find the next available log file name
    counter=1
    log_file="${log_dir}/output_resume.log"
    while [[ -f "$log_file" ]]; do
        counter=$((counter + 1))
        # Build the log file name with the right number of _resume
        log_file="${log_dir}/output"
        for ((i=1; i<=counter; i++)); do
            log_file="${log_file}_resume"
        done
        log_file="${log_file}.log"
    done
    
    # Append the same number of _resume to run_name
    for ((i=1; i<=counter; i++)); do
        run_name="${run_name}_resume"
    done
else
    # If not resuming, just use output.log
    log_file="${log_dir}/output.log"
    mkdir -p "${log_dir}"
fi


python scripts/run/train.py --config ${config_path} --run-name "${run_name}" --log-dir "${log_dir}" 2>&1 | tee "${log_file}"
