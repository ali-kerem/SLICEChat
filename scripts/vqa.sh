cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PYTHONPATH="${PWD}:${PYTHONPATH}"

source .venv/bin/activate

python scripts/run/vqa.py \
    --ckpt_dir /path/to/slideinstruct_ckpt/checkpoints/checkpoint \
    --manifest-path data/generated/slidebench_vqa_tcga.json \
    --data_dir /path/to/wsi-tensors/ \
    --output-path /path/to/results.json
