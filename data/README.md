# SLICEChat dataset preparation

This directory contains reproducible builders for the training and evaluation manifests used by SLICEChat, along with a guide to extracting patch embeddings with TRIDENT and converting them into the feature and coordinate tensors used for training and inference.

## Frozen upstream sources

| Source | Revision | Annotations |
| --- | --- | --- |
| [SlideChat](https://huggingface.co/datasets/General-Medical-AI/SlideChat/tree/975a73561ab8ff93455f9d6f2e5a571f940dc9c7) | `975a73561ab8ff93455f9d6f2e5a571f940dc9c7` | `SlideInstruct_train_stage1_caption.json`, `SlideInstruct_train_stage2_vqa.json`, `SlideBench-VQA-TCGA.csv`, `SlideBench-VQA-BCNB.csv` |
| [WSI-LLaVA](https://github.com/XinhengLyu/WSI-LLaVA/tree/7a98bbcb8807a4396a4b971fc7ffaf05cb8f7f90) | `7a98bbcb8807a4396a4b971fc7ffaf05cb8f7f90` | All 16 `dataset/*_train.json` files; the 11 `dataset/*_test.json` files without `choice` or `TF` in their names |

These are the same revisions used by SLICEChat-encoder. Caption and VQA training annotations are merged into one manifest per dataset. The `choice`/`TF` exclusion applies to **evaluation only**: we build only the open-ended part of WSI-Bench for evaluation. `source_checksums.json` records SHA-256 hashes of the annotation files and their pinned download URLs.

## Filename mapping and curated exclusions

`slidechat_filename_map.csv` resolves SlideChat's short barcodes to full slide filename stems. The mapping and the following two lists are the same as SLICEChat-encoder's `data/` directory:

- `dataset_exclusions/missing_mpp_slide_ids.txt`: 69 identifiers.
- `dataset_exclusions/invalid_slide_ids.txt`: 205 identifiers.

`invalid_slide_ids.txt` contains identifiers for slides that were unavailable from GDC when we prepared the training data. This does not mean that those slides never existed or cannot become available later.

### Exclusion counts

| Dataset | Subset | Missing MPP: WSIs / pairs | Invalid-list: WSIs / pairs |
| --- | --- | ---: | ---: |
| SlideInstruct | Caption | 29 / 29 | 0 / 0 |
| SlideInstruct | VQA | 29 / 1,284 | 0 / 0 |
| SlideBench | TCGA | 8 / 21 | 0 / 0 |
| SlideBench | BCNB | 0 / 0 | 0 / 0 |
| WSI-Bench | Train | 67 / 1,260 | 205 / 3,608 |
| WSI-Bench | Test | 5 / 5 | 13 / 33 |

## Annotation corrections

`annotation_corrections.json` stores one-based source row indices, hashes of the original records, and replacement conversations. The builder verifies a record's hash before applying its replacement; `null` marks the record for removal. Row indices refer to entries in the upstream JSON arrays, not file line numbers.

### SlideInstruct VQA corrections

Categories described in Section 4 of our paper's supplementary material:

| Issue | Affected records | Corrected and retained | Removed |
| --- | ---: | ---: | ---: |
| Missing option letters | 155 | 155 | 0 |
| Missing option text | 3 | 1 | 2 |
| Answer not present in the options | 8 | 0 | 8 |
| Empty assistant answer | 4 | 0 | 4 |
| **Total** | **170** | **156** | **14** |

The missing-letter fixes comprise six manual replacements and 149 automatic repairs. Automatic repairs also expand letter-only answers to include the selected option's text; this is not applied to other records.

Of the three questions with labels but no option text, two had letter-only answers and were removed. The third retained its textual answer after removing the empty labels and answer-letter prefix. Records with answers outside the listed options or empty assistant answers are removed.

After slide exclusions and corrections, **174,436 VQA records** and **4,152 captions** form the **178,588-record** training manifest.

### WSI-LLaVA training normalization

| Difference from the pinned source | Records |
| --- | ---: |
| One trailing double quote removed from a question | 111 |
| One trailing double quote removed from an answer | 10 |
| Conversation reordered from assistant/user to user/assistant | 1 |
| Record with an empty assistant answer removed | 1 |

This gives **122 changed records and one removal**. Quote cleanup removes one literal trailing `"`, not JSON escaping; other text and whitespace remain unchanged. The reordered conversation is entry 1815 of `Staging_choice_train.json`. The empty-answer removal is entry 13042 of `Diagnosis_choice_train.json`; the slide's other records are kept.

WSI-LLaVA caption prefixes are preserved for MLLM training, unlike the encoder's CLIP-caption preprocessing.

## Build

From the repository root, activate the project's environment and run the builder (Python 3.11+, standard library only):

```bash
source .venv/bin/activate
python data/create_mllm_manifests.py
```

By default, the builder downloads missing SlideChat and WSI-LLaVA annotations from the pinned URLs, checks their hashes, and creates the five manifests below. Source files are cached under `data/source_annotations/<source>/<revision>/<upstream-path>`.

To use local annotations and a different output directory:

```bash
python data/create_mllm_manifests.py \
    --slidechat-source-dir /path/to/slidechat-annotations \
    --wsillava-source-dir /path/to/wsillava-annotations \
    --no-download-source-files \
    --output-dir /path/to/manifests
```

Both source overrides default to `None`, using the revision-based cache above. Custom roots contain the upstream paths: SlideChat files directly under its root, and WSI-LLaVA files under `dataset/`. Use `--source-dir` to relocate the shared cache instead. Downloads are enabled by default; existing sources are hash-checked and never overwritten. Outputs default to `data/generated` and are protected unless `--overwrite-outputs` is passed.

| Generated manifest | Pairs | WSIs |
| --- | ---: | ---: |
| `generated/slideinstruct_train.json` | 178,588 | 4,152 |
| `generated/wsibench_train.json` | 172,305 | 9,596 |
| `generated/slidebench_vqa_tcga.json` | 1,473 | 583 |
| `generated/slidebench_vqa_bcnb.json` | 7,274 | 1,058 |
| `generated/wsibench.json` | 2,799 | 739 |

`generated/audit.json` summarizes source and retained record counts for each subset, excluded slide identifiers and their pair counts, annotation corrections, and source revisions.

## Extracting Patch Embeddings

Use [TRIDENT](https://github.com/mahmoodlab/Trident) to segment the WSIs, extract non-overlapping 512-pixel patches at 20x, and encode them with CONCH v1.5. Install TRIDENT by following its repository instructions, then run this command from the TRIDENT repository:

```bash
python run_batch_of_slides.py \
    --task all \
    --wsi_dir /path/to/wsis \
    --job_dir /path/to/trident_output \
    --overlap 0 \
    --patch_size 512 \
    --mag 20 \
    --patch_encoder conch_v15
```

For BCNB, set the source image resolution to **0.25 μm/pixel** using TRIDENT's `--custom_list_of_wsis` argument. Create `slides.csv` with every image's actual filename, including its extension:

```csv
wsi,mpp
1.jpg,0.25
2.jpg,0.25
```

Pass `--custom_list_of_wsis slides.csv` to the TRIDENT extraction command. See the [BCNB resolution discussion](https://github.com/bupt-ai-cz/BALNMP/issues/1).

The relevant HDF5 files are written to:

```text
/path/to/trident_output/20x_512px_0px_overlap/features_conch_v15/
```

### Convert HDF5 outputs to PyTorch tensors

```bash
source .venv/bin/activate
python data/h5_to_pt.py \
    --h5-files-dir /path/to/trident_output/20x_512px_0px_overlap/features_conch_v15 \
    --save-dir /path/to/wsi-tensors \
    --patch-size 512 \
    --num-workers 4
```

| Argument | Description |
| --- | --- |
| `--h5-files-dir` | Directory containing TRIDENT's `.h5` files. |
| `--save-dir` | Output root for `features/` and `coords/`. |
| `--patch-size` | Patch size in pixels used during extraction. |
| `--num-workers` | Number of parallel conversion processes. |

The converter preserves patch features, converts pixel coordinates to patch-grid indices, and normalizes their common grid spacing, following the encoder's coordinate convention. `--patch-size` must match the value passed to TRIDENT.

## Data format

The converter creates one feature file and one coordinate file per slide:

```text
/path/to/wsi-tensors/
├── features/
│   ├── TCGA-xx-xxxx-xx.{uuid}.pt
│   └── TCGA-yy-yyyy-yy.{uuid}.pt
└── coords/
    ├── TCGA-xx-xxxx-xx.{uuid}.pt
    └── TCGA-yy-yyyy-yy.{uuid}.pt
```

Feature tensors have shape `[num_patches, feature_dim]`; coordinates have shape `[num_patches, 2]` in the same patch order. The feature dimension must match the pretrained slide encoder.

Each JSON record uses the complete filename stem, without a directory or `.pt` suffix. `WSIDataset` resolves both tensors beneath the configured `data_dir`. Training manifests contain user/assistant conversations:

```json
[
  {
    "filename": "TCGA-xx-xxxx-xx.{uuid}",
    "project_id": "TCGA-LUAD",
    "conversations": [
      {"role": "user", "content": "Describe the findings in this slide."},
      {"role": "assistant", "content": "Pathology description."}
    ]
  }
]
```

Evaluation manifests contain user-only conversations, with the reference answer stored separately:

```json
[
  {
    "filename": "TCGA-xx-xxxx-xx.{uuid}",
    "question": "What is the diagnosis?",
    "answer": "Reference diagnosis.",
    "conversations": [
      {"role": "user", "content": "What is the diagnosis?"}
    ]
  }
]
```
