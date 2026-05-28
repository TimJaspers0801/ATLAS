"""
ATLAS-120k class definitions.

Modelled after the Cityscapes label taxonomy. Each entry in `atlas_classes`
describes one original palette class (pixel value in the machine_masks PNGs).
The `train_id` field gives the consolidated training class used during model
training (0 = background / excluded). Classes with `train_id == 0` are either
excluded from evaluation or merged into the background.

`train_classes` is the complementary list of 30 training classes, each
carrying its own display colour for visualising model predictions.

Usage
-----
    from atlas120k import atlas_classes, train_classes

    # Map an original mask to training IDs
    import numpy as np
    lut = np.array([c.train_id for c in atlas_classes], dtype=np.uint8)
    train_mask = lut[original_mask]

    # Get the colour palette for training-class visualisation
    palette = {c.train_id: c.color for c in train_classes}
"""

from collections import namedtuple

# ---------------------------------------------------------------------------
# Original class taxonomy (pixel value → full class description)
# ---------------------------------------------------------------------------

AtlasClass = namedtuple(
    "AtlasClass",
    [
        "name",      # original class name
        "id",        # pixel value in machine_masks PNG  (0 = background)
        "train_id",  # consolidated training class ID    (0 = background / excluded)
        "category",  # anatomical region
        "color",     # RGB colour for mask visualisation (original palette)
    ],
)

# fmt: off
atlas_classes = [
    #                   name                        id   train_id  category      color
    AtlasClass("Background",                         0,   0,       "background", (  0,   0,   0)),

    # ── Abdomen ─────────────────────────────────────────────────────────────
    AtlasClass("Tools/camera",                       1,   1,       "abdomen",    (255, 255, 255)),
    AtlasClass("Vein (major)",                       2,   2,       "abdomen",    (  0,   0, 255)),
    AtlasClass("Artery (major)",                     3,   3,       "abdomen",    (255,   0,   0)),
    AtlasClass("Nerve (major)",                      4,   4,       "abdomen",    (255, 255,   0)),
    AtlasClass("Small intestine",                    5,   5,       "abdomen",    (  0, 255,   0)),
    AtlasClass("Colon/rectum",                       6,   6,       "abdomen",    (  0, 200, 100)),
    AtlasClass("Abdominal wall",                     7,   7,       "abdomen",    (200, 150, 100)),
    AtlasClass("Diaphragm",                          8,   8,       "abdomen",    (250, 150, 100)),
    AtlasClass("Omentum",                            9,   9,       "abdomen",    (255, 200, 100)),  # → Fat
    AtlasClass("Aorta",                             10,   3,       "abdomen",    (180,   0,   0)),  # → Artery
    AtlasClass("Vena cava",                         11,   2,       "abdomen",    (  0,   0, 180)),  # → Vein
    AtlasClass("Liver",                             12,  10,       "abdomen",    (150, 100,  50)),
    AtlasClass("Cystic duct",                       13,  11,       "abdomen",    (  0, 255, 255)),  # → Bile/lymph duct
    AtlasClass("Gallbladder",                       14,  12,       "abdomen",    (  0, 200, 255)),
    AtlasClass("Hepatic vein",                      15,   2,       "abdomen",    (  0, 100, 255)),  # → Vein
    AtlasClass("Hepatic ligament",                  16,  13,       "abdomen",    (255, 150,  50)),
    AtlasClass("Cystic plate",                      17,  14,       "abdomen",    (255, 220, 200)),
    AtlasClass("Stomach",                           18,  15,       "abdomen",    (200, 100, 200)),
    AtlasClass("Ductus choledochus",                19,  11,       "abdomen",    (144, 238, 144)),  # → Bile/lymph duct
    AtlasClass("Mesenterium",                       20,   9,       "abdomen",    (247, 255,   0)),  # → Fat
    AtlasClass("Ductus hepaticus",                  21,  11,       "abdomen",    (255, 206,  27)),  # → Bile/lymph duct
    AtlasClass("Spleen",                            22,  16,       "abdomen",    (200,   0, 200)),
    AtlasClass("Uterus",                            23,  17,       "abdomen",    (255,   0, 150)),
    AtlasClass("Ovary",                             24,  18,       "abdomen",    (255, 100, 200)),
    AtlasClass("Oviduct",                           25,  19,       "abdomen",    (200, 100, 255)),

    # ── RARP ────────────────────────────────────────────────────────────────
    AtlasClass("Prostate",                          26,  20,       "rarp",       (150,   0, 100)),
    AtlasClass("Urethra",                           27,  21,       "rarp",       (255, 200, 255)),
    AtlasClass("Ligated plexus",                    28,  22,       "rarp",       (150, 100,  75)),
    AtlasClass("Seminal vesicles",                  29,  23,       "rarp",       (200,   0, 150)),
    AtlasClass("Catheter",                          30,  24,       "rarp",       (100, 100, 100)),  # → Non anatomical
    AtlasClass("Bladder",                           31,  25,       "rarp",       (255, 150, 255)),
    AtlasClass("Kidney",                            32,   0,       "rarp",       (100, 200, 255)),  # → Background

    # ── Thorax ──────────────────────────────────────────────────────────────
    AtlasClass("Lung",                              33,  26,       "thorax",     (150, 200, 255)),
    AtlasClass("Airway (bronchus/trachea)",         34,  27,       "thorax",     (  0, 150, 255)),
    AtlasClass("Esophagus",                         35,  28,       "thorax",     (255, 100, 100)),
    AtlasClass("Pericardium",                       36,  29,       "thorax",     (200, 200, 255)),
    AtlasClass("V azygos",                          37,   2,       "thorax",     (100, 100, 255)),  # → Vein
    AtlasClass("Thoracic duct",                     38,  11,       "thorax",     (  0, 255, 150)),  # → Bile/lymph duct
    AtlasClass("Nerves",                            39,   4,       "thorax",     (255, 255, 100)),  # → Nerve

    # ── Non-anatomical / excluded ────────────────────────────────────────────
    AtlasClass("Ureter",                            40,   0,       "special",    (150, 150, 150)),  # → Background
    AtlasClass("Non anatomical structures",         41,  24,       "special",    ( 50,  50,  50)),  # → Non anatomical
    AtlasClass("Excluded frames",                   42,   0,       "special",    (  0,   0,   0)),  # → Background

    # ── Additional structures (mapped to background) ─────────────────────────
    AtlasClass("Mesocolon",                         43,   0,       "abdomen",    (173, 216, 230)),  # → Background
    AtlasClass("Adrenal gland",                     44,   0,       "abdomen",    (255, 140,   0)),  # → Background
    AtlasClass("Pancreas",                          45,   0,       "abdomen",    (223,   3, 252)),  # → Background
    AtlasClass("Duodenum",                          46,   0,       "abdomen",    (  0,  80, 100)),  # → Background
]
# fmt: on

# ---------------------------------------------------------------------------
# Training class taxonomy (train_id → name + display colour)
# ---------------------------------------------------------------------------

TrainClass = namedtuple("TrainClass", ["name", "train_id", "color"])

# fmt: off
train_classes = [
    #                          name                       train_id  color
    TrainClass("Background",                              0,        (  0,   0,   0)),
    TrainClass("Tools/camera",                            1,        (255, 255, 255)),
    TrainClass("Vein",                                    2,        (  0,   0, 255)),
    TrainClass("Artery",                                  3,        (255,   0,   0)),
    TrainClass("Nerve",                                   4,        (255, 255,   0)),
    TrainClass("Small intestine",                         5,        (  0, 255,   0)),
    TrainClass("Colon/rectum",                            6,        (  0, 200, 100)),
    TrainClass("Abdominal wall",                          7,        (200, 150, 100)),
    TrainClass("Diaphragm",                               8,        (250, 150, 100)),
    TrainClass("Fat",                                     9,        (255, 200, 100)),
    TrainClass("Liver",                                  10,        (150, 100,  50)),
    TrainClass("Bile/lymph duct",                        11,        (  0, 255, 255)),
    TrainClass("Gallbladder",                            12,        (  0, 200, 255)),
    TrainClass("Hepatic ligament",                       13,        (255, 150,  50)),
    TrainClass("Cystic plate",                           14,        (255, 220, 200)),
    TrainClass("Stomach",                                15,        (200, 100, 200)),
    TrainClass("Spleen",                                 16,        (200,   0, 200)),
    TrainClass("Uterus",                                 17,        (255,   0, 150)),
    TrainClass("Ovary",                                  18,        (255, 100, 200)),
    TrainClass("Oviduct",                                19,        (200, 100, 255)),
    TrainClass("Prostate",                               20,        (150,   0, 100)),
    TrainClass("Urethra",                                21,        (255, 200, 255)),
    TrainClass("Ligated plexus",                         22,        (150, 100,  75)),
    TrainClass("Seminal vesicles",                       23,        (200,   0, 150)),
    TrainClass("Non anatomical",                         24,        ( 50,  50,  50)),
    TrainClass("Bladder",                                25,        (255, 150, 255)),
    TrainClass("Lung",                                   26,        (150, 200, 255)),
    TrainClass("Airway (bronchus/trachea)",              27,        (  0, 150, 255)),
    TrainClass("Esophagus",                              28,        (255, 100, 100)),
    TrainClass("Pericardium",                            29,        (200, 200, 255)),
]
# fmt: on

# ---------------------------------------------------------------------------
# Convenience look-up tables
# ---------------------------------------------------------------------------

# Original palette: id → RGB
palette = {c.id: c.color for c in atlas_classes}

# Training palette: train_id → RGB
train_palette = {c.train_id: c.color for c in train_classes}

# Mapping from original ID to train_id as a flat list (index = original id)
# Suitable for use as a numpy LUT: train_mask = id_to_train_id[original_mask]
id_to_train_id = [c.train_id for c in sorted(atlas_classes, key=lambda c: c.id)]
