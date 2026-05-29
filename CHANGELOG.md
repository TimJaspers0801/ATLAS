# Changelog

All notable changes to the ATLAS-120k dataset and this repository are documented here.

Version numbers follow [Semantic Versioning](https://semver.org/):
- **MAJOR** — structural changes incompatible with prior versions (e.g. changed mask format, renamed splits)
- **MINOR** — new content added in a backwards-compatible way (e.g. new videos, new annotated classes)
- **PATCH** — backwards-compatible annotation corrections (e.g. fixed masks, corrected class labels)

The dataset version is recorded in [`dataset_info.json`](dataset_info.json) and in the HuggingFace release metadata.

---

## [0.1.0] — 2026

### Initial public release

- 100 surgical videos across 14 minimally invasive procedures (laparoscopic and robot-assisted)
- 502 annotated clips, 121,018 frames at 15 fps
- 42 anatomical classes; 30-class consolidated training taxonomy
- Splits: 70 train / 10 val / 20 test videos
- Segmentation masks in single-channel PNG format (pixel value = original class ID)
- Per-video `clip_index.json` files listing exact annotated frame numbers
- Per-video `surgical_index` files encoding the manually curated surgical frame range
