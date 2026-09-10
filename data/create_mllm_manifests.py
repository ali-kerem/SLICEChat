"""Rebuild SLICEChat annotations; never downloads slides or inspects MPP."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

# Default paths are relative to this file, not the current directory.
DATA_DIR = Path(__file__).resolve().parent
SOURCE_DIR = DATA_DIR / "source_annotations"
OUTPUT_DIR = DATA_DIR / "generated"

# Use the same revisions as SLICEChat-encoder for both training and evaluation.
SLIDECHAT_REVISION = "975a73561ab8ff93455f9d6f2e5a571f940dc9c7"
WSILLAVA_REVISION = "7a98bbcb8807a4396a4b971fc7ffaf05cb8f7f90"
TASKS = (
    "1_Report", "Diagnosis_choice", "Global_Morphology_Description", "Grading",
    "Histological_Typing", "Key_Diagnostic_Description", "Molecular_Subtyping_choice",
    "Molecular_Subtyping", "Morphology_choice", "Prognosis_TF_question", "Prognosis",
    "Regional_Structure_Description", "Specific_Feature_Description", "Staging_choice",
    "Staging", "Treatment_Planing",
)
SLIDECHAT_FILES = (
    "SlideInstruct_train_stage1_caption.json", "SlideInstruct_train_stage2_vqa.json",
    "SlideBench-VQA-TCGA.csv", "SlideBench-VQA-BCNB.csv",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR,
                        help="Base annotation cache (default: data/source_annotations).")
    parser.add_argument("--slidechat-source-dir", type=Path, default=None,
                        help="SlideChat annotation root; defaults to <source-dir>/slidechat/<revision>.")
    parser.add_argument("--wsillava-source-dir", type=Path, default=None,
                        help="WSI-LLaVA root containing dataset/; defaults to <source-dir>/wsillava/<revision>.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Manifest output directory (default: data/generated).")
    parser.add_argument("--download-source-files", action=argparse.BooleanOptionalAction,
                        default=True, help="Download missing annotations (default: enabled).")
    parser.add_argument("--overwrite-outputs", action=argparse.BooleanOptionalAction,
                        default=False, help="Replace existing generated manifests (default: disabled).")
    return parser.parse_args(argv)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_ids(path):
    return {line.strip().lower() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def source_file(name, revision, family, lock, args):
    key = f"{family}/{revision}/{name}"
    source_dir = {"slidechat": args.slidechat_source_dir, "wsillava": args.wsillava_source_dir}[family]
    root = args.source_dir / family / revision if source_dir is None else source_dir
    destination = root / name
    if not destination.is_file():
        if args.download_source_files:
            with urlopen(lock[key]["url"], timeout=120) as response:
                payload = response.read()
        else:
            raise FileNotFoundError(f"Missing annotation: {destination}")
        if hashlib.sha256(payload).hexdigest() != lock[key]["sha256"]:
            raise ValueError(f"Source hash mismatch: {key}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    with destination.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != lock[key]["sha256"]:
        raise ValueError(f"Source hash mismatch: {destination}; use the documented frozen file")
    return destination


def image_info(row):
    image = row["image"]
    if isinstance(image, list):
        if len(image) != 1:
            raise ValueError("Expected exactly one WSI per annotation")
        image = image[0]
    path = Path(image)
    if path.suffix not in {".pt", ".csv"}:
        raise ValueError(f"Unexpected source filename extension: {image}")
    project = path.parent.name
    return path.stem, project if project.startswith("TCGA-") else f"TCGA-{project}"


def conversations(row):
    roles = {"human": "user", "gpt": "assistant"}
    return [{"role": roles[turn["from"]],
             "content": turn["value"].replace("<image>\n", "").replace("\n<image>", "")}
            for turn in row["conversations"]]


class ExclusionAudit:
    def __init__(self, missing, invalid):
        self.missing = missing
        self.invalid = invalid
        self.counts = {"missing_mpp": Counter(), "invalid": Counter()}
        self.source_pairs = 0
        self.annotation_drops = 0
        self.annotation_changes = 0

    def exclude(self, filename):
        self.source_pairs += 1
        key = filename.lower()
        for reason, ids in (("missing_mpp", self.missing), ("invalid", self.invalid)):
            if key in ids:
                self.counts[reason][key] += 1
                return True
        return False

    def report(self, rows):
        return {
            "source_pairs": self.source_pairs,
            **{reason: {"wsis": len(counts), "pairs": sum(counts.values()),
                        "slides": dict(sorted(counts.items()))}
               for reason, counts in self.counts.items()},
            "annotation_drops": self.annotation_drops,
            "annotation_changes": self.annotation_changes,
            "retained_pairs": len(rows),
            "retained_wsis": len({row["filename"].lower() for row in rows}),
            "empty_assistant_turns": sum(
                turn["role"] == "assistant" and not turn["content"].strip()
                for row in rows for turn in row["conversations"]
            ),
        }


def corrected_conversations(row, index, patches, audit):
    patch = patches.get(str(index))
    if patch is None:
        return conversations(row)
    if row_hash(row) != patch["sha256"]:
        raise ValueError(f"Annotation correction does not match source row {index}")
    result = patch["conversations"]
    if result is None:
        audit.annotation_drops += 1
    else:
        audit.annotation_changes += 1
    return result


def training_rows(path, mapping, missing, invalid, corrections):
    audit = ExclusionAudit(missing, invalid)
    output = []
    missing_by_barcode = {}
    if mapping is not None:
        for key in missing:
            missing_by_barcode.setdefault(key.split(".")[0], []).append(key)
    for index, row in enumerate(read_json(path), 1):
        filename, project = image_info(row)
        # Missing-MPP short barcodes may intentionally have no filename mapping.
        if mapping is not None:
            if filename.lower() in missing_by_barcode:
                matches = missing_by_barcode[filename.lower()]
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous excluded barcode: {filename}")
                filename = matches[0]
            else:
                filename = mapping[filename.lower()]
        if audit.exclude(filename):
            continue
        conv = corrected_conversations(row, index, corrections.get(path.name, {}), audit)
        if conv is not None:
            output.append({"filename": filename, "project_id": project, "conversations": conv})
    return output, audit


def slidebench_rows(path, mapping, missing, invalid):
    audit = ExclusionAudit(missing, invalid)
    output = []
    missing_by_barcode = {key.split(".")[0]: key for key in missing}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            filename = row["Slide"]
            if mapping is not None:
                key = filename.lower()
                filename = missing_by_barcode[key] if key in missing_by_barcode else mapping[key]
            if audit.exclude(filename):
                continue
            options = [f"{label}) {row[label]}" for label in "ABCD" if row.get(label, "").strip()]
            question = row["Question"] + (" " + " ".join(options) if options else "")
            result = {"filename": filename, "question": question, "answer": row["Answer"],
                      "conversations": [{"role": "user", "content": question}],
                      "category": row.get("Broad Category", row.get("Task"))}
            if "Tumor" in row:
                result["project_id"] = "TCGA-" + row["Tumor"]
            output.append(result)
    return output, audit


def validate_rows(rows, evaluation):
    for row in rows:
        filename = row["filename"]
        if not filename or Path(filename).name != filename or filename.endswith(".pt"):
            raise ValueError(f"Expected a filename stem: {filename!r}")
        conv = row["conversations"]
        expected = ["user"] if evaluation else ["user", "assistant"]
        if [turn["role"] for turn in conv] != expected:
            raise ValueError(f"Unexpected conversation roles for {filename}")
        if any(not isinstance(t["content"], str) for t in conv):
            raise ValueError(f"Non-string conversation text for {filename}")
        if not conv[0]["content"].strip():
            raise ValueError(f"Empty question for {filename}")
        if not evaluation and not conv[1]["content"].strip():
            raise ValueError(f"Empty assistant answer for {filename}")
        if evaluation and (not isinstance(row["answer"], str) or not row["answer"].strip()):
            raise ValueError(f"Missing reference answer for {filename}")


def main():
    args = parse_args()
    names = ["slideinstruct_train.json", "wsibench_train.json", "slidebench_vqa_tcga.json",
             "slidebench_vqa_bcnb.json", "wsibench.json", "audit.json"]
    if not args.overwrite_outputs:
        existing = [str(args.output_dir / name) for name in names if (args.output_dir / name).exists()]
        if existing:
            raise FileExistsError(f"Outputs already exist: {existing}; choose another --output-dir or use --overwrite-outputs")
    lock = read_json(DATA_DIR / "source_checksums.json")
    corrections = read_json(DATA_DIR / "annotation_corrections.json")
    missing = read_ids(DATA_DIR / "dataset_exclusions/missing_mpp_slide_ids.txt")
    invalid = read_ids(DATA_DIR / "dataset_exclusions/invalid_slide_ids.txt")
    with (DATA_DIR / "slidechat_filename_map.csv").open() as handle:
        mapping = {row["slide_id"].lower(): row["filename"] for row in csv.DictReader(handle)}
    reports = {"settings": {"slidechat_revision": SLIDECHAT_REVISION,
                            "wsillava_revision": WSILLAVA_REVISION}}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write(name, rows, evaluation=False):
        validate_rows(rows, evaluation)
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"{name}: {len(rows):,} pairs", flush=True)

    merged = []
    for name, subset in zip(SLIDECHAT_FILES[:2], ("Caption", "VQA")):
        path = source_file(name, SLIDECHAT_REVISION, "slidechat", lock, args)
        rows, audit = training_rows(path, mapping, missing, invalid, corrections)
        reports[f"SlideInstruct {subset}"] = audit.report(rows)
        merged.extend(rows)
    write("slideinstruct_train.json", merged)
    del merged, rows
    for subset in ("TCGA", "BCNB"):
        path = source_file(f"SlideBench-VQA-{subset}.csv", SLIDECHAT_REVISION, "slidechat", lock, args)
        rows, audit = slidebench_rows(path, mapping if subset == "TCGA" else None,
                                      missing if subset == "TCGA" else set(),
                                      invalid if subset == "TCGA" else set())
        reports[f"SlideBench {subset}"] = audit.report(rows)
        write(f"slidebench_vqa_{subset.lower()}.json", rows, evaluation=True)

    merged = []
    combined = ExclusionAudit(missing, invalid)
    for task in TASKS:
        path = source_file(f"dataset/{task}_train.json", WSILLAVA_REVISION, "wsillava", lock, args)
        rows, audit = training_rows(path, None, missing, invalid, corrections)
        merged.extend(rows)
        combined.source_pairs += audit.source_pairs
        combined.annotation_changes += audit.annotation_changes
        combined.annotation_drops += audit.annotation_drops
        for reason in combined.counts:
            combined.counts[reason].update(audit.counts[reason])
    reports["WSI-Bench Train"] = combined.report(merged)
    write("wsibench_train.json", merged)
    del merged, rows

    rows = []
    audit = ExclusionAudit(missing, invalid)
    for task in TASKS:
        if "choice" in task.lower() or "tf" in task.lower():
            continue
        path = source_file(f"dataset/{task}_test.json", WSILLAVA_REVISION, "wsillava", lock, args)
        for row in read_json(path):
            filename, project = image_info(row)
            if audit.exclude(filename):
                continue
            conv = conversations(row)
            question = next(t["content"].strip() for t in conv if t["role"] == "user")
            answer = next(t["content"].strip() for t in conv if t["role"] == "assistant")
            rows.append({"filename": filename, "project_id": project, "question": question,
                         "answer": answer, "category": task,
                         "conversations": [{"role": "user", "content": question}]})
    reports["WSI-Bench Test"] = audit.report(rows)
    write("wsibench.json", rows, evaluation=True)
    with (args.output_dir / "audit.json").open("w") as handle:
        json.dump(reports, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
