<div align="center">

# ATLAS-120k

**Surgical Anatomy Recognition with Context Learning using Foundation Representations**

[![Paper](https://img.shields.io/badge/Paper-MICCAI_2026-blue?style=flat)](https://arxiv.org/)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-HuggingFace-ffcc00?style=flat)](https://huggingface.co/TimJaspersTue/datasets/atlas120k)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<img src="figures/atlas120k.png" alt="ATLAS-120k dataset examples" width="100%">

</div>

---

ATLAS-120k is a large-scale clip-level semantic segmentation dataset for surgical anatomy recognition in minimally invasive surgery (MIS). It comprises over **120,000 annotated frames** from **100 surgical videos** spanning **14 procedures** and **42 anatomical classes**, covering both laparoscopic and robot-assisted surgery.

This repository provides the tools to download and process the raw video data. Annotations (segmentation masks) are distributed separately on [HuggingFace](https://huggingface.co/TimJaspersTue/datasets/atlas120k).

## Repository Ecosystem

The full ATLAS project spans multiple repositories:

| Repository | Description |
|---|---|
| **This repo** | Dataset download and processing scripts |
| [ATLAS-Interactive](https://github.com/rlpddejong/ATLAS-Interactive) | Annotation platform, interactive segmentation tools and annotation models |
| [atlas-bench](https://github.com/TimJaspers0801/atlas-bench) | Benchmark experiments and ATLAS model implementations |
| [SurgeNetDINO](https://github.com/rlpddejong/SurgeNetDINO) | Pretrained DINOv1/v2/v3 surgical foundation backbones |
| [SurgeNet](https://github.com/TimJaspers0801/SurgeNet) | SurgeNet pretraining dataset used for surgical foundation models |

## Dataset

### Statistics

| Dataset | Procedures | Classes | Videos | Clips | Frames | Laparoscopic | Robot-assisted |
|---|---|---|---|---|---|---|---|
| Endoscapes-Seg50 | 1 | 6 | 50 | — | 493 | ✓ | |
| CholecSeg8k | 1 | 12 | 17 | — | 8,080 | ✓ | |
| DSAD | 1 | 11 | 32 | — | 14,625 | | ✓ |
| **ATLAS-120k** | **14** | **42** | **100** | **503** | **120,776** | **✓** | **✓** |

### Procedures

| | | | |
|---|---|---|---|
| Adrenalectomy | Appendectomy | Cholecystectomy | Colectomy |
| Esophagectomy | Gastric surgery | Gastrojejunostomy | Hemicolectomy |
| Laparoscopic anterior resection (LAR) | Liver resection | RARP | Rectopexy |
| Sigmoid resection | Splenectomy | | |

### Anatomical Classes

42 classes including: liver, gallbladder, cystic duct, hepatic ligament, cystic plate, ductus choledochus, ductus hepaticus, stomach, small intestine, colon/rectum, abdominal wall, diaphragm, omentum, aorta, vena cava, artery (major), vein (major), nerve (major), spleen, pancreas, duodenum, kidney, bladder, ureter, uterus, ovary, prostate, seminal vesicles, adrenal gland, mesocolon, mesenterium, V. azygos, esophagus, pericardium, airway (bronchus/trachea), lung, catheter, and tools/camera.

## Getting Started

### Prerequisites

```bash
pip install -r download/requirements.txt
```

`ffmpeg` must be installed and available on `PATH` — it is used for fps conversion and frame extraction. See [ffmpeg.org](https://ffmpeg.org/download.html).

For YouTube downloads requiring authentication, provide a cookies file — see the [yt-dlp cookies guide](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp).

### Step 1 — Download annotations from HuggingFace

Download the annotation archive from [HuggingFace](https://huggingface.co/TimJaspersTue/datasets/atlas120k) and extract it to a local directory, e.g. `atlas120k/`. This contains the segmentation masks and per-video clip index files, but **not** the images (which you provide by downloading the YouTube videos in Step 2).

```
atlas120k/
  train/  val/  test/
    └── <procedure>/
          └── <video_folder>/
                ├── clip_index.json        ← frame lists per clip
                └── <clip>/
                      └── machine_masks/   ← segmentation masks
```

### Step 2 — Download raw videos from YouTube

YouTube links for each procedure are in `data/youtube_links/`. Add the URLs for each procedure to the corresponding `.txt` file (one URL per line), then run:

```bash
python download/download_videos.py \
    --links_dir  data/youtube_links \
    --output_dir raw_data \
    --workers    4 \
    --cookies    /path/to/cookies.txt   # optional
```

Videos are saved to `raw_data/<procedure>/<youtube_id>.<ext>`.

### Step 3 — Process videos into the dataset

This script converts each raw video to 15 fps and extracts the exact frames listed in the clip index files, saving them into the `images/` subfolder of each clip inside the annotation download.

```bash
python download/process_atlas120k.py \
    --raw_data_dir raw_data \
    --atlas_dir    atlas120k \
    --workers      4
```

To process only specific procedures or splits:

```bash
python download/process_atlas120k.py \
    --raw_data_dir raw_data \
    --atlas_dir    atlas120k \
    --splits       test \
    --procedures   cholecystectomy appendectomy
```

## Dataset Structure

After completing all steps the dataset looks like:

```
raw_data/                               ← raw YouTube downloads (Step 2)
  cholecystectomy/
    4FpUGXO9mzg.mp4
    J5bg8KTYrw0.mp4
    ...
  appendectomy/
    ...

atlas120k/                              ← HuggingFace download + extracted frames
  train/
    cholecystectomy/
      4FpUGXO9mzg/
        clip_index.json                 ← frame numbers per clip (from HF)
        clip_0001/
          images/                       ← extracted by process_atlas120k.py
            frame_000000.jpg
            frame_000001.jpg
            ...
          machine_masks/                ← from HuggingFace
            frame_000000.png
            frame_000001.png
            ...
      J5bg8KTYrw0_ROBOT/
        ...
  val/
    ...
  test/
    ...
```

Frame filenames encode the **global frame index in the 15 fps video** (e.g. `frame_000964.jpg` is frame 964 of the downsampled video). Frames within a clip may be non-contiguous — the `clip_index.json` files list the exact frames included.

Segmentation masks are single-channel PNG files where each pixel value is a class index. The class-to-index mapping is provided in `class_index.json` in the HuggingFace release.

## Clip Index Format

Each video directory contains a `clip_index.json` file:

```json
{
  "folder_name": "J5bg8KTYrw0_ROBOT",
  "youtube_id": "J5bg8KTYrw0",
  "procedure": "cholecystectomy",
  "split": "train",
  "is_robot": true,
  "frame_digits": 6,
  "clips": {
    "clip_0001": [964, 965, 966, ...],
    "clip_0002": [1004, 1005, ..., 1264]
  }
}
```

`frame_digits` is the zero-padding width used in all frame filenames for that video (either 4 or 6). The clip index files are generated by `download/generate_clip_index.py` (dataset authors only) and distributed with the HuggingFace release.

## Ethics

**Sources.** The videos in ATLAS-120k were sourced from publicly available surgical videos on YouTube, originally collected as part of the [GSViT dataset](https://github.com/SamuelSchmidgall/GSViT). Our data collection prioritized videos from medical institutions and surgeons, aiming to maximize compliance with professional consent standards.

**Curation and identifiability.** The dataset consists exclusively of intraoperative endoscopic footage. Prior to annotation, all clips were manually reviewed to remove out-of-body views, patient-identifiable content (e.g., faces, text overlays), and non-surgical content. Annotations were performed by three surgical research fellows under the supervision of three experienced surgeons (each with more than 10 years of experience), and the first frame of every clip was independently reviewed by at least one experienced surgeon for correctness. To the best of our knowledge, patients are not identifiable by current methods from endoscopic video alone.

**Regulatory.** In many jurisdictions, including the EU, the sharing of non-identifiable data acquired for routine medical purposes does not require explicit patient consent and is consistent with the GDPR.

**Opt-out.** If you are the owner of a source video and wish to have it removed from the dataset, please open an issue in this repository or contact us directly. Additionally, if a source video is removed from YouTube, the corresponding items will be removed from future releases of the dataset.

## Acknowledgements

The video sources for ATLAS-120k were drawn from the **[GSViT dataset](https://github.com/SamuelSchmidgall/GSViT)**. We thank the authors for making their collection publicly available.

The download pipeline in this repository was inspired by the tooling from **[SurgLaVi](https://github.com/aperezr20/SurgLaVi)** and **[LEMON](https://github.com/visurg-ai/LEMON)**.

## License

Dataset annotations: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Code in this repository: [MIT License](LICENSE)

## Citation

If you use ATLAS-120k in your research, please cite:

```bibtex
@inproceedings{atlas2026miccai,
  title     = {Surgical Anatomy Recognition with Context Learning using Foundation Representations},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2026},
}
```
