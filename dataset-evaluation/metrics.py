import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def compute_class_metrics(pred, gt, class_id, ignore_index=255):
    """
    Computes IoU and Dice for a specific class ID in one image.
    Returns (iou, dice) or (None, None) if class not present in GT.
    
    Args:
        ignore_index: Pixels with this value in GT are excluded from computation
    """
    # Create masks excluding ignore_index
    valid_mask = (gt != ignore_index)
    p = (pred == class_id) & valid_mask
    g = (gt == class_id) & valid_mask

    if not np.any(g):
        return None, None

    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    sum_pixels = p.sum() + g.sum()

    iou = intersection / union if union > 0 else 0.0
    dice = (2. * intersection) / sum_pixels if sum_pixels > 0 else 0.0

    return iou, dice
def mask_to_rle(binary_mask):
    """
    binary_mask: HxW numpy array (0,1)
    """
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")  # json serializable
    return rle


class SegmentationAPEvaluator:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.predictions = []
        self.image_id = 0
        self.ann_id = 0

    def add_frame(self, gt_mask, pred_mask, score):
        """
        gt_mask, pred_mask: HxW binary numpy arrays
        score: confidence score for prediction
        """
        h, w = gt_mask.shape
        img_id = self.image_id
        self.image_id += 1

        # image entry
        self.images.append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": f"{img_id}.png"
        })

        # GT annotation
        gt_rle = mask_to_rle(gt_mask)
        area = gt_mask.sum()

        self.annotations.append({
            "id": self.ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": gt_rle,
            "area": float(area),
            "bbox": [0, 0, w, h],
            "iscrowd": 0
        })
        self.ann_id += 1

        # Prediction
        pred_rle = mask_to_rle(pred_mask)

        self.predictions.append({
            "image_id": img_id,
            "category_id": 1,
            "segmentation": pred_rle,
            "score": float(score)
        })

    def evaluate(self):
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": 1, "name": "object"}]
        }
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(self.predictions)

        coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "AP": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
        }
