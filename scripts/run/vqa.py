from pathlib import Path
from dotenv import load_dotenv

# Load Hugging Face settings before libraries read them at import time.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import json
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.mllms.mllm_factory import MLLMFactory
from src.datasets import DatasetFactory
from src.data_utils.collators import CollatorFactory
from src.data_utils.preprocessors import PreprocessorFactory
from src.utils.checkpoint import get_checkpoint_config


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument("--manifest-path", type=str, required=True, help="Path to a generated SlideBench or WSI-Bench evaluation manifest")
    parser.add_argument("--data_dir", type=str, required=True, help="Tensor root containing features/ and coords/ directories")
    parser.add_argument("--output-path", type=Path, required=True, help="Exact output JSON path (parent directories are created if needed)")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = get_checkpoint_config(args.ckpt_dir)


    preprocessor = PreprocessorFactory.create(
        preprocessor_type=config["model"]["modality"],
        tokenizer_name_or_path=config["model"]["llm_name_or_path"],
        **config["data"]["preprocessor_args"]
    )

    model = MLLMFactory.from_pretrained(
        args.ckpt_dir,
        placeholder_tokens=preprocessor.get_placeholder_token_name_to_ids(),
        config=config,
    )
    model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    dataset = DatasetFactory.create(
        dataset_name="wsi",
        labels_path=args.manifest_path,
        data_dir=args.data_dir,
        preprocessor=preprocessor,
        system_prompt=config["data"]["dataset_args"].get("system_prompt"),
        return_labels=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=CollatorFactory.create(modality="slide"),
    )

    output_records = []

    for idx, data_dict in enumerate(tqdm(dataloader, desc="Evaluating VQA")):
        # Sequential single-sample batches preserve the metadata/output association.
        sample = dataset.dataset[idx]
        # The shared collator returns variable-length token sequences as a list.
        data_dict["input_ids"] = data_dict["input_ids"][0].unsqueeze(0)
        data_dict = {k: v.to(device="cuda") for k, v in data_dict.items()}

        data_dict["features"] = data_dict["features"].to(dtype=torch.bfloat16)

        with torch.no_grad():
            outputs = model.generate(**data_dict, max_new_tokens=500, do_sample=False)
            out_text = preprocessor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            out_text = out_text.lstrip("!")


            record = dict(sample)
            record["model_response"] = out_text


            output_records.append(record)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=2)
    print(f"Saved outputs ({len(output_records)}) to {args.output_path}")

if __name__ == "__main__":
    main()
