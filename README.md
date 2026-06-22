<div align="center">

# ATLAS-120k

**Surgical Anatomy Recognition with Context Learning using Foundation Representations**

[![Paper](https://img.shields.io/badge/Paper-MICCAI_2026-blue?style=flat)](https://arxiv.org/)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-HuggingFace-ffcc00?style=flat)](https://huggingface.co/datasets/TimJaspersTue/ATLAS-120k)
[![Version](https://img.shields.io/badge/dataset-v0.1.0-green?style=flat)](CHANGELOG.md)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<img src="figures/atlas120k.png" alt="ATLAS-120k dataset examples" width="100%">

</div>

---

ATLAS-120k is a large-scale clip-level semantic segmentation dataset for surgical anatomy recognition in minimally invasive surgery (MIS). It comprises over **120,000 annotated frames** from **100 surgical videos** spanning **14 procedures** and **42 anatomical classes**, covering both laparoscopic and robot-assisted surgery.

This repository provides the tools to download and process the raw video data. Annotations (segmentation masks) are distributed separately on [HuggingFace](https://huggingface.co/datasets/TimJaspersTue/ATLAS-120k).

## Repository Ecosystem

The full ATLAS project spans multiple repositories:

| Repository | Description |
|---|---|
| **This repo** | Dataset download and processing scripts |
| [ATLAS-interactive](https://github.com/rlpddejong/ATLAS-Interactive) | Annotation platform, interactive segmentation tools and annotation models |
| [ATLAS-model](https://github.com/rlpddejong/ATLAS-model) | ATLAS model implementation and training code | 
| [ATLAS-bench](https://github.com/TimJaspers0801/ATLAS-bench-public) | Benchmark experiments of various SOTA models |
| [SurgeNetDINO](https://github.com/rlpddejong/SurgeNetDINO) | Pretrained DINOv1/v2/v3 surgical foundation backbones |
| [SurgeNet](https://github.com/TimJaspers0801/SurgeNet) | SurgeNet pretraining dataset used for surgical foundation models |

## Dataset

### Statistics

| Dataset | Procedures | Classes | Videos | Clips | Frames | Laparoscopic | Robot-assisted |
|---|---|---|---|---|---|---|---|
| Endoscapes-Seg50 | 1 | 6 | 50 | — | 493 | ✓ | |
| CholecSeg8k | 1 | 12 | 17 | — | 8,080 | ✓ | |
| DSAD | 1 | 11 | 32 | — | 14,625 | | ✓ |
| **ATLAS-120k** | **14** | **42** | **100** | **502** | **121,018** | **✓** | **✓** |

### Procedures
| | | | |
|---|---|---|---|
| Adrenalectomy | Appendectomy | Cholecystectomy | Colectomy |
| Esophagectomy | Gastric surgery | Gastrojejunostomy | Hemicolectomy |
| Laparoscopic anterior resection (LAR) | Liver resection | RARP | Rectopexy |
| Sigmoid resection | Splenectomy |  |  |

### Anatomical Classes

42 classes including: liver, gallbladder, cystic duct, hepatic ligament, cystic plate, ductus choledochus, ductus hepaticus, stomach, small intestine, colon/rectum, abdominal wall, diaphragm, omentum, aorta, vena cava, artery (major), vein (major), nerve (major), spleen, pancreas, duodenum, kidney, bladder, ureter, uterus, ovary, prostate, seminal vesicles, adrenal gland, mesocolon, mesenterium, V. azygos, esophagus, pericardium, airway (bronchus/trachea), lung, catheter, and tools/camera.

Full class definitions — including the consolidated 30-class training taxonomy and colour palette — are provided in [`atlas120k_tools/classes.py`](atlas120k_tools/classes.py).

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

`ffmpeg` must be installed and available on `PATH` — it is used for fps conversion and frame extraction. See [ffmpeg.org](https://ffmpeg.org/download.html). Note: only `ffmpeg` itself is required; `ffprobe` is not needed as video metadata is read via OpenCV.

A JavaScript runtime (**[Deno](https://deno.com/)** recommended, or Node.js / Bun) must also be on `PATH`. yt-dlp uses it to solve YouTube's challenge that protects the video stream URLs; without it downloads return only storyboard images and fail with `Only images are available`. Install Deno (e.g. `winget install DenoLand.Deno` on Windows, or see [deno.com](https://deno.com/)) and confirm with `deno --version`.

**YouTube cookies are required.** The source videos are surgical content that YouTube marks as age-restricted. You must be signed in to YouTube in your browser, then authenticate the downloader using one of two methods:

**Option A — read directly from your browser (easiest, no export needed):**
```bash
python download/download_videos.py \
    --links_dir data/youtube_links \
    --output_dir raw_data \
    --cookies-from-browser chrome   # or: firefox, edge, safari
```

**Option B — export a cookies.txt file once and reuse it:**
```bash
# 1. Install a browser extension such as "Get cookies.txt LOCALLY" (Chrome/Firefox)
#    and export cookies for youtube.com → save as cookies.txt
# 2. Pass the file to the downloader:
python download/download_videos.py \
    --links_dir data/youtube_links \
    --output_dir raw_data \
    --cookies cookies.txt
```

Option A is simpler. Option B is useful on headless servers where no browser is available. See the [yt-dlp cookies guide](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) for more details.

### Step 1 — Download annotations from HuggingFace

Download the annotations from [HuggingFace](https://huggingface.co/datasets/TimJaspersTue/ATLAS-120k) into a local directory called `atlas120k/`. This contains the segmentation masks and per-video clip index files, but **not** the images (which you provide by downloading the YouTube videos in Step 2).

The dataset is **gated**: first sign in on HuggingFace, open the [dataset page](https://huggingface.co/datasets/TimJaspersTue/ATLAS-120k), and accept the terms. You will also need an access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (a **Read** token, or a fine-grained token with "Read access to contents of all public gated repos" enabled).

**Recommended — `git clone` (most reliable for the ~120k small files).** Because the dataset is many tiny mask files, the HuggingFace REST API (used by `hf download`) quickly hits a rate limit of 1000 requests per 5 minutes and stalls. `git clone` transfers everything over the git protocol in batches and avoids this entirely. Make sure [git-lfs](https://git-lfs.com/) is installed (`git lfs install`), then:

```bash
git clone https://USER:TOKEN@huggingface.co/datasets/TimJaspersTue/ATLAS-120k atlas120k
```

Replace `USER` with your HuggingFace username and `TOKEN` with your `hf_...` access token. For large repos you can skip the LFS blobs during clone and pull them in a single batched pass afterwards:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://USER:TOKEN@huggingface.co/datasets/TimJaspersTue/ATLAS-120k atlas120k
cd atlas120k && git lfs pull
```

After cloning, strip the embedded credentials from the local repo config:

```bash
git -C atlas120k remote set-url origin https://huggingface.co/datasets/TimJaspersTue/ATLAS-120k
```

**Alternative — `hf` CLI** (simpler, but slower and rate-limit prone on a dataset this size; `huggingface_hub` is included in `requirements.txt`). Authenticate with `hf auth login` first, then keep `--max-workers` low to stay under the API rate limit:

```bash
hf download TimJaspersTue/ATLAS-120k --repo-type dataset --local-dir atlas120k --max-workers 2
```

It resumes interrupted transfers, so you can safely re-run it if it stops.

```
atlas120k/
  train/  val/  test/
    └── <procedure>/
          └── <video_folder>/
                ├── clip_index.json        ← annotated frame lists per clip
                └── <clip>/
                      └── machine_masks/   ← segmentation masks (pixel value = class ID)
```

### Step 2 — Download raw videos from YouTube

YouTube links for each procedure are provided in `data/youtube_links/` (one `.txt` file per procedure, one URL per line). Run:

```bash
python download/download_videos.py \
    --links_dir  data/youtube_links \
    --output_dir raw_data \
    --workers    4 \
    --cookies    /path/to/cookies.txt   # optional
```

Videos are saved to `raw_data/<procedure>/<youtube_id>.<ext>`.

> **Note:** 3 videos are no longer publicly available on YouTube. For these videos both the segmentation masks and the processed frames are included directly in the HuggingFace release — no download is required. The raw full videos are not available. These 3 videos will be replaced with new ones in a future dataset release.

### Step 3 — Process videos into the dataset

This script converts each raw video to 15 fps, applies the surgical frame filter (skipping non-surgical content identified during dataset curation), and extracts the exact annotated frames into the `images/` subfolder of each clip.

```bash
python download/process_atlas120k.py \
    --raw_data_dir       raw_data \
    --atlas_dir          atlas120k \
    --surgical_index_dir data/surgical_index \
    --workers            4
```

To process only specific procedures or splits:

```bash
python download/process_atlas120k.py \
    --raw_data_dir       raw_data \
    --atlas_dir          atlas120k \
    --surgical_index_dir data/surgical_index \
    --splits             test \
    --procedures         cholecystectomy appendectomy
```

To also save **all** surgical frames per video (reproducing the full 15 fps surgical-only video):

```bash
python download/process_atlas120k.py \
    --raw_data_dir           raw_data \
    --atlas_dir              atlas120k \
    --surgical_index_dir     data/surgical_index \
    --save_surgical_frames \
    --surgical_frames_dir    atlas120k_full_videos \
    --workers                4
```

## Repository Structure

```
data/
  youtube_links/               ← one .txt per procedure with YouTube URLs
    cholecystectomy.txt
    appendectomy.txt
    ...
  surgical_index/              ← per-video surgical frame range (included in this repo)
    cholecystectomy/
      4FpUGXO9mzg.json
      ...
    appendectomy/
    ...

download/
  download_videos.py           ← Step 2: download YouTube videos
  process_atlas120k.py         ← Step 3: extract frames with surgical filtering
  dataset_stats.py             ← print per-split statistics and cross-check dataset_info.json
  generate_clip_index.py       ← (authors only) generate clip_index.json from annotated dataset
  generate_surgical_index.py   ← (authors only) generate surgical_index from curated full videos

dataset-evaluation/
  dataset_evaluation.py        ← evaluate_model() and evaluate_atlas_temporal() entry points
  metrics.py                   ← compute_class_metrics() (IoU/Dice) and SegmentationAPEvaluator

atlas120k_tools/               ← Python package (import as `atlas120k_tools`)
  __init__.py
  classes.py                   ← class definitions, colour palette, train ID mapping

atlas120k/                     ← dataset download target (Step 1; not in this repo)

requirements.txt               ← Python dependencies for all steps
```

## Dataset Structure

After completing all steps the full dataset looks like:

```
raw_data/                               ← raw YouTube downloads (Step 2)
  cholecystectomy/
    4FpUGXO9mzg.mp4
    J5bg8KTYrw0.mp4
    ...

atlas120k/                              ← HuggingFace download + extracted frames (Step 3)
  train/
    cholecystectomy/
      4FpUGXO9mzg/
        clip_index.json                 ← annotated frame numbers per clip
        clip_0001/
          images/                       ← extracted by process_atlas120k.py
            frame_000000.jpg
            frame_000001.jpg
            ...
          machine_masks/                ← segmentation masks (from HuggingFace)
            frame_000000.png
            frame_000001.png
            ...
      J5bg8KTYrw0_ROBOT/
        ...
  val/  test/
    ...
```

Frame filenames encode the **global frame index in the 15 fps video** (e.g. `frame_000964.jpg` is frame 964 of the downsampled video). Frames within a clip may be non-contiguous — the `clip_index.json` files list the exact frames included.

Segmentation masks are single-channel PNG files where each pixel value is a class ID. See [`atlas120k_tools/classes.py`](atlas120k_tools/classes.py) for the full ID-to-name mapping and colour palette.

## Reference Files

### Clip Index (`clip_index.json`)

Located at `<split>/<procedure>/<video_folder>/clip_index.json` in the HuggingFace download. Lists the exact frame numbers belonging to every annotated clip.

```json
{
  "folder_name": "J5bg8KTYrw0_ROBOT",
  "youtube_id": "J5bg8KTYrw0",
  "procedure": "cholecystectomy",
  "split": "train",
  "is_robot": true,
  "frame_digits": 6,
  "clips": {
    "clip_0001": [964, 965, 966, "..."],
    "clip_0002": [1004, 1005, "...", 1264]
  }
}
```

`frame_digits` is the zero-padding width used in all frame filenames for that video (4 or 6).

### Surgical Index (`data/surgical_index/<procedure>/<youtube_id>.json`)

Included in this repository. Records which portions of each raw video contain surgical content, so that non-surgical frames (pre-operative setup, post-operative, text overlays, etc.) are automatically skipped during processing.

```json
{
  "youtube_id": "6X7BRo4hNt8",
  "procedure": "esophagectomy",
  "start_frame": 77,
  "end_frame": 12244,
  "excluded_ranges": [[9543, 9963]],
  "total_surgical_frames": 11747
}
```

`excluded_ranges` lists `[from, to]` (inclusive) ranges of non-surgical frames within the surgical window. For videos without removals, `excluded_ranges` is empty and the surgical content runs contiguously from `start_frame` to `end_frame`.

### Class Definitions (`atlas120k_tools/classes.py`)

Defines the full 46-entry original class taxonomy and the consolidated 30-class training taxonomy, modelled after the Cityscapes label format:

```python
from atlas120k_tools import atlas_classes, train_classes
import numpy as np

# Convert an original mask (pixel = class ID 0–46) to training IDs (0–29)
from atlas120k_tools.classes import id_to_train_id
lut = np.array(id_to_train_id, dtype=np.uint8)
train_mask = lut[original_mask]

# Colour palette for visualisation
from atlas120k_tools.classes import train_palette  # {train_id: (R, G, B)}
```

## Evaluation

The `dataset-evaluation/` module provides ready-to-use evaluation functions for models trained on ATLAS-120k.

### Metrics

| Metric | Description |
|---|---|
| **mIoU** | Mean Intersection over Union across foreground classes |
| **mDice** | Mean Dice coefficient across foreground classes |
| **AP / AP50 / AP75** | COCO-style Average Precision (binary, frame-level) |
| **mVC₁₂ / mVC₂₄** | Mean Video Consistency over sliding windows of 12 and 24 frames |

### Standard model

```python
from dataset-evaluation.dataset_evaluation import evaluate_model

results = evaluate_model(
    model=model,
    dataloader=test_loader,
    device=device,
    num_classes=30,     # number of foreground classes (background excluded)
    compute_ap=True,    # set False to skip AP for faster evaluation
)
# results: mIoU, Dice, AP, AP50, AP75, per_class_iou, per_class_dice, mVC_8, mVC_12, mVC_24
```

### ATLAS temporal model (with query propagation)

```python
from dataset-evaluation.dataset_evaluation import evaluate_atlas_temporal

results = evaluate_atlas_temporal(
    model=model,
    test_loader=test_loader,   # must yield frames in clip order, batch size = 1
    device=device,
    num_classes=30,
    use_query_propagation=True,
    compute_ap=True,
)
```

The dataloader must supply batches with keys `"image"`, `"mask"`, `"procedure"`, `"video"`, and `"clip"`. Frames within a clip must be yielded in temporal order; query embeddings are automatically reset at clip boundaries.

## Versioning

The dataset and this repository follow [Semantic Versioning](https://semver.org/). The current version is **v0.1.0**.

| Bump | When to use |
|---|---|
| **PATCH** `1.0.x` | Annotation corrections — fixed masks, corrected class labels, updated clip boundaries |
| **MINOR** `1.x.0` | New content added — additional videos, new annotated classes, new splits |
| **MAJOR** `x.0.0` | Breaking changes — renamed splits, changed mask format or class ID scheme |

Each release updates three places:
1. [`VERSION`](VERSION) and [`dataset_info.json`](dataset_info.json) in this repository
2. The version badge at the top of this README and in [`CHANGELOG.md`](CHANGELOG.md)
3. A new tagged GitHub release and a new HuggingFace dataset revision

To check which version of the dataset you have locally:

```python
import json
info = json.load(open("atlas120k/dataset_info.json"))  # included in HF release
print(info["version"])
```

See [`CHANGELOG.md`](CHANGELOG.md) for the full history of changes.

## Ethics

**Sources.** The videos in ATLAS-120k were sourced from publicly available surgical videos on YouTube, originally collected as part of the [GSViT dataset](https://github.com/SamuelSchmidgall/GSViT). Our data collection prioritized videos from medical institutions and surgeons, aiming to maximize compliance with professional consent standards.

**Curation and identifiability.** The dataset consists exclusively of intraoperative endoscopic footage. Prior to annotation, all clips were manually reviewed to remove out-of-body views, patient-identifiable content (e.g., faces, text overlays), and non-surgical content. Annotations were performed by three surgical research fellows under the supervision of three experienced surgeons (each with more than 10 years of experience), and the first frame of every clip was independently reviewed by at least one experienced surgeon for correctness. To the best of our knowledge, patients are not identifiable by current methods from endoscopic video alone.

**Regulatory.** In many jurisdictions, including the EU, the sharing of non-identifiable data acquired for routine medical purposes does not require explicit patient consent and is consistent with the GDPR.

**Opt-out.** If you are the owner of a source video and wish to have it removed from the dataset, please open an issue in this repository or contact us directly. Additionally, if a source video is removed from YouTube, the corresponding items will be removed from future releases of the dataset.

## Acknowledgements

The video sources for ATLAS-120k were drawn from the **[GSViT dataset](https://github.com/SamuelSchmidgall/GSViT)**. We thank the authors for making their collection publicly available.

The download pipeline in this repository was inspired by the tooling from **[SurgLaVi](https://github.com/aperezr20/SurgLaVi)** and **[LEMON](https://github.com/visurg-ai/LEMON)**.
We also acknowledge **[YouTube-VOS](https://youtube-vos.org/)** as inspiration for creating a large-scale video dataset setting, adapted here to the surgical domain.

## License

Dataset annotations: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Code in this repository: [MIT License](LICENSE)

## Citation

If you use ATLAS-120k in your research, please cite:

```bibtex
@inproceedings{TBA,
}
```
