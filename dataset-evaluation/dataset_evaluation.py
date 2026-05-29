import torch
import gc
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from typing import List, Optional
from .metrics import compute_class_metrics, SegmentationAPEvaluator


def compute_mVC_for_clip(
    preds_list: List[np.ndarray],
    gts_list: List[np.ndarray],
    n: int,
    background_label: int = 0,
    skip_empty_windows: bool = True,
) -> Optional[float]:
    """
    Compute VC_n for a single clip (sliding windows).
    Returns clip-level VC_n (mean over windows) or None if the clip
    had no valid windows (all windows had empty gt_common after excluding background).

    Args:
        preds_list: List of 2D integer label maps (H x W), in temporal order
        gts_list: List of 2D integer label maps (H x W), in temporal order
        n: Window size (number of consecutive frames)
        background_label: Label value for background class
        skip_empty_windows: If True, skip windows with no valid gt pixels

    Returns:
        Temporal consistency (VC) metric for the clip or None if no valid windows
    """
    C = len(preds_list)
    if C < n:
        return None

    window_vcs = []
    total_windows = C - n + 1

    for i in range(total_windows):
        wp = np.stack(preds_list[i : i + n], axis=0)  # (n, H, W)
        wg = np.stack(gts_list[i : i + n], axis=0)  # (n, H, W)

        # Find pixels where GT is consistent across all frames in window
        gt_equal = np.all(wg == wg[0:1], axis=0)  # (H, W)
        gt_label_first = wg[0]  # (H, W)
        gt_common_nonbg = gt_equal & (gt_label_first != background_label)
        denom = int(gt_common_nonbg.sum())

        if denom == 0:
            if skip_empty_windows:
                continue  # skip window as invalid
            else:
                window_vcs.append(0.0)
                continue

        # Find pixels where prediction is also consistent
        pred_common = np.all(wp == wp[0:1], axis=0)  # (H, W)
        num = int(np.logical_and(gt_common_nonbg, pred_common).sum())

        ratio = float(num) / float(denom)
        window_vcs.append(ratio)

    if len(window_vcs) == 0:
        return None

    return float(np.mean(window_vcs))


def evaluate_model(model, dataloader, device, num_classes, compute_ap=False):
    """
    Args:
        model: The model to evaluate.
        dataloader: The dataloader for evaluation.
        device: Device to use (cuda or cpu).
        num_classes: Number of classes.
        compute_ap: If True, compute AP metrics (slower). If False, skip AP. Default False.
    """
    model.eval()
    
    ignore_index = 255  # Standard ignore value used in Cityscapes and other datasets

    # Track scores per class across the entire dataset
    # format: {class_id: [score_img1, score_img2, ...]}
    class_ious = defaultdict(list)
    class_dices = defaultdict(list)

    # Track predictions and GTs per clip for temporal consistency metrics
    clip_frames = defaultdict(lambda: {"preds": [], "gts": []})

    # AP logic (only if compute_ap=True)
    clip_ap = {}
    current_clip = None
    ap_evaluator = None

    classes_to_eval = range(1, num_classes + 1)  # Skip background 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Periodic memory cleanup every 1000 iterations
            if batch_idx > 0 and batch_idx % 1000 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Resize logits to match ground truth dimensions using bilinear interpolation
            # This must happen BEFORE argmax (same as in training code)
            if probs.shape[-2:] != gt_masks.shape[-2:]:
                probs = torch.nn.functional.interpolate(
                    probs,
                    size=gt_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            preds = torch.argmax(probs, dim=1, keepdim=True)

            for i in range(len(images)):
                # Safe extraction of clip ID
                try:
                    clip_id = f"{batch['procedure'][i]}/{batch['video'][i]}/{batch['clip'][i]}"
                except (KeyError, TypeError, IndexError):
                    # Fallback clip ID if metadata unavailable
                    clip_id = f"unknown_{batch_idx}_{i}"
                
                # --- AP Handling (only if compute_ap=True) ---
                if compute_ap:
                    if current_clip != clip_id:
                        if ap_evaluator is not None:
                            clip_ap[current_clip] = ap_evaluator.evaluate()
                        current_clip = clip_id
                        ap_evaluator = SegmentationAPEvaluator()

                pred_np = preds[i, 0].cpu().numpy()
                gt_np = gt_masks[i].cpu().numpy().squeeze()
                
                # Store for temporal metrics
                clip_frames[clip_id]["preds"].append(pred_np)
                clip_frames[clip_id]["gts"].append(gt_np)
                
                gt_np_masked = gt_np

                # --- Metric Collection (The part you wanted) ---
                for c in classes_to_eval:
                    iou_c, dice_c = compute_class_metrics(pred_np, gt_np_masked, c, ignore_index=ignore_index)

                    if iou_c is not None:
                        class_ious[c].append(iou_c)
                        class_dices[c].append(dice_c)

                # Binary AP logic (only if compute_ap=True)
                if compute_ap:
                    gt_binary = (gt_np > 0).astype(np.uint8)
                    pred_binary = (pred_np > 0).astype(np.uint8)
                    
                    # Compute per-frame confidence score: average probability of predicted class in predicted mask
                    # Move to CPU to avoid large GPU tensor allocation
                    probs_np = probs[i].cpu().numpy()  # Shape: (num_classes, H, W)
                    preds_np = preds[i, 0].cpu().numpy()  # Shape: (H, W) - class index per pixel
                    
                    # Use advanced indexing to get prob of predicted class for each pixel
                    H, W = preds_np.shape
                    h_indices, w_indices = np.mgrid[0:H, 0:W]
                    pred_class_probs = probs_np[preds_np, h_indices, w_indices]  # Shape: (H, W)
                    pred_mask = (pred_np > 0)  # Mask of non-background predictions
                    
                    if pred_mask.sum() > 0:
                        score = pred_class_probs[pred_mask].mean()
                    else:
                        # If no mask predicted, use max probability as fallback
                        score = probs_np.max()
                    
                    ap_evaluator.add_frame(gt_binary, pred_binary, score)
            
            # Delete tensors to free memory immediately
            del images, gt_masks, outputs, probs, preds

        if compute_ap and ap_evaluator is not None:
            clip_ap[current_clip] = ap_evaluator.evaluate()
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()

    # --- FINAL DATASET AGGREGATION ---

    final_per_class_iou = {}
    final_per_class_dice = {}

    for c in classes_to_eval:
        if len(class_ious[c]) > 0:
            final_per_class_iou[c] = np.mean(class_ious[c])
            final_per_class_dice[c] = np.mean(class_dices[c])
        else:
            final_per_class_iou[c] = 0.0
            final_per_class_dice[c] = 0.0

    # The Final Mean Scores
    mIoU = np.mean(list(final_per_class_iou.values()))
    mDice = np.mean(list(final_per_class_dice.values()))

    # AP Metrics
    AP_total = np.mean([v["AP"] for v in clip_ap.values()]) if clip_ap else 0.0
    AP_50 = np.mean([v["AP50"] for v in clip_ap.values()]) if clip_ap else 0.0
    AP_75 = np.mean([v["AP75"] for v in clip_ap.values()]) if clip_ap else 0.0

    # Compute temporal consistency (mVC) for different window sizes
    window_sizes = [8, 12, 24]
    mvc_metrics = {}
    
    for window_n in window_sizes:
        clip_vcs = []
        for clip_id, frames_data in clip_frames.items():
            vc = compute_mVC_for_clip(
                frames_data["preds"],
                frames_data["gts"],
                n=window_n,
                background_label=0,
                skip_empty_windows=True,
            )
            if vc is not None:
                clip_vcs.append(vc)
        
        if clip_vcs:
            mvc_metrics[f"mVC_{window_n}"] = np.mean(clip_vcs)
        else:
            mvc_metrics[f"mVC_{window_n}"] = 0.0

    # --- Print Detailed Report ---
    print("\n" + "=" * 50)
    print(f"{'Class ID':<10} | {'IoU':<10} | {'Dice':<10}")
    print("-" * 50)
    for c in classes_to_eval:
        print(f"Class {c:<4} | {final_per_class_iou[c]:.4f}     | {final_per_class_dice[c]:.4f}")
    print("-" * 50)
    print(f"{'OVERALL':<10} | {mIoU:.4f}     | {mDice:.4f}")
    print(f"{'mAP':<10} | {AP_total:.4f}")
    print("-" * 50)
    print("Temporal Consistency (mVC - Mean across clips):")
    for window_n in window_sizes:
        print(f"  mVC_{window_n:<2} | {mvc_metrics[f'mVC_{window_n}']:.4f}")
    print("=" * 50 + "\n")

    return {
        "mIoU": mIoU,
        "Dice": mDice,
        "AP": AP_total,
        "AP50": AP_50,
        "AP75": AP_75,
        "per_class_iou": final_per_class_iou,
        "per_class_dice": final_per_class_dice,
        "mVC_8": mvc_metrics["mVC_8"],
        "mVC_12": mvc_metrics["mVC_12"],
        "mVC_24": mvc_metrics["mVC_24"],
    }


def evaluate_atlas_temporal(
    model,
    test_loader,
    device,
    num_classes,
    use_query_propagation=True,
    compute_ap=False,
):
    """
    Evaluate ATLAS model with temporal processing on a clip basis.
    
    Processes each clip frame-by-frame in an online manner:
    - Query embeddings are propagated between frames within the same clip
    - Query embeddings are reset when moving to a new clip
    - Batch size should be 1 for proper temporal state management
    
    Args:
        model: ATLAS model with temporal capabilities
        test_loader: DataLoader yielding individual frames with clip metadata
        device: torch device
        num_classes: number of segmentation classes
        use_query_propagation: whether to propagate query embeddings between frames
        compute_ap: If True, compute AP metrics (slower). If False, skip AP. Default False.
    
    Returns:
        Dictionary of metrics (mIoU, Dice, AP, AP50, AP75, per_class metrics)
    """
    model.eval()
    
    ignore_index = 255
    
    # Track metrics per class
    class_ious = defaultdict(list)
    class_dices = defaultdict(list)
    
    # Track predictions and GTs per clip for temporal consistency metrics
    clip_frames = defaultdict(lambda: {"preds": [], "gts": []})
    
    # Track AP per clip (only if compute_ap=True)
    clip_ap = {}
    current_clip = None
    ap_evaluator = None

    classes_to_eval = range(1, num_classes + 1)  # Skip background 0

    # Track query embeddings for propagation
    prev_query_embed = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating ATLAS temporal")):
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)

            # Periodic memory cleanup every 1000 iterations
            if batch_idx > 0 and batch_idx % 1000 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Get clip information
            procedure = batch["procedure"][0] if isinstance(batch["procedure"], list) else batch["procedure"]
            video = batch["video"][0] if isinstance(batch["video"], list) else batch["video"]
            clip_id = batch["clip"][0] if isinstance(batch["clip"], list) else batch["clip"]

            clip_key = f"{procedure}/{video}/{clip_id}"

            # Reset query embeddings when entering a new clip
            if current_clip != clip_key:
                if compute_ap and ap_evaluator is not None:
                    clip_ap[current_clip] = ap_evaluator.evaluate()
                current_clip = clip_key
                if compute_ap:
                    ap_evaluator = SegmentationAPEvaluator()
                prev_query_embed = None  # Reset queries for new clip
            
            # Forward pass with query propagation
            outputs = model(
                images,
                prev_query_embed=prev_query_embed if use_query_propagation else None,
                return_query_embedding=use_query_propagation,
            )
            
            # Unpack outputs
            if use_query_propagation:
                mask_logits_per_block, class_logits_per_block, procedure_logits_per_block, query_embed = outputs
                prev_query_embed = query_embed  # Store for next frame within same clip
            else:
                mask_logits_per_block, class_logits_per_block, procedure_logits_per_block = outputs
            
            # Get predictions from final block
            mask_logits = mask_logits_per_block[-1]  # (B, num_q, H, W)
            class_logits = class_logits_per_block[-1]  # (B, num_q, num_classes+1)
            
            # Convert query-level predictions to per-pixel logits
            # Shape: (B, num_classes, H, W) - logits at model output resolution
            per_pixel_logits = torch.einsum(
                "bqhw, bqc -> bchw",
                mask_logits.sigmoid(),
                class_logits.softmax(dim=-1)[..., :-1]  # Exclude background class
            )
            
            # Resize logits to match ground truth dimensions using bilinear interpolation
            # This must happen BEFORE argmax (same as in training code)
            if per_pixel_logits.shape[-2:] != gt_masks.shape[-2:]:
                per_pixel_logits = torch.nn.functional.interpolate(
                    per_pixel_logits,
                    size=gt_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Get predictions
            preds = torch.argmax(per_pixel_logits, dim=1)  # (B, H, W)
            
            # Evaluate each frame in the batch
            classes_to_eval = range(1, num_classes + 1)  # Skip background 0
            
            for b in range(images.shape[0]):
                pred_np = preds[b].cpu().numpy()
                gt_np = gt_masks[b].cpu().numpy().squeeze()
                
                # Store for temporal metrics
                clip_frames[clip_key]["preds"].append(pred_np)
                clip_frames[clip_key]["gts"].append(gt_np)
                
                # --- Compute per-class metrics (IoU, Dice) ---
                for c in classes_to_eval:
                    iou_c, dice_c = compute_class_metrics(
                        pred_np, gt_np, c, ignore_index=ignore_index
                    )
                    
                    if iou_c is not None:
                        class_ious[c].append(iou_c)
                        class_dices[c].append(dice_c)
                
                # --- Compute AP metrics (binary, only if compute_ap=True) ---
                if compute_ap and ap_evaluator is not None:
                    gt_binary = (gt_np > 0).astype(np.uint8)
                    pred_binary = (pred_np > 0).astype(np.uint8)
                    
                    # Compute per-frame confidence score
                    probs_np = class_logits[b].softmax(dim=-1)[:, :-1].max(dim=-1)[0].cpu().numpy()  # (num_q,)
                    mask_prob = mask_logits[b].sigmoid().cpu().numpy()  # (num_q, H, W)
                    
                    # Average confidence across queries and spatial dimensions
                    if mask_prob.sum() > 0:
                        score = (mask_prob * probs_np[:, None, None]).sum() / mask_prob.sum()
                    else:
                        score = 0.5
                    
                    ap_evaluator.add_frame(gt_binary, pred_binary, score)
            
            # Delete tensors to free memory immediately
            del images, gt_masks, mask_logits_per_block, class_logits_per_block, procedure_logits_per_block
            del per_pixel_logits, preds
    
    # Finalize last clip's AP (only if compute_ap=True)
    if compute_ap and ap_evaluator is not None:
        clip_ap[current_clip] = ap_evaluator.evaluate()
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # --- Compute final metrics ---
    final_per_class_iou = {}
    final_per_class_dice = {}
    
    for c in classes_to_eval:
        if len(class_ious[c]) > 0:
            final_per_class_iou[c] = np.mean(class_ious[c])
            final_per_class_dice[c] = np.mean(class_dices[c])
        else:
            final_per_class_iou[c] = 0.0
            final_per_class_dice[c] = 0.0
    
    # Compute overall metrics
    mIoU = np.mean(list(final_per_class_iou.values()))
    mDice = np.mean(list(final_per_class_dice.values()))
    
    # AP Metrics (set to 0.0 if not computed)
    if compute_ap:
        AP_total = np.mean([v["AP"] for v in clip_ap.values()]) if clip_ap else 0.0
        AP_50 = np.mean([v["AP50"] for v in clip_ap.values()]) if clip_ap else 0.0
        AP_75 = np.mean([v["AP75"] for v in clip_ap.values()]) if clip_ap else 0.0
    else:
        AP_total = 0.0
        AP_50 = 0.0
        AP_75 = 0.0
    
    # Compute temporal consistency (mVC) for different window sizes
    window_sizes = [8, 12, 24]
    mvc_metrics = {}
    
    for window_n in window_sizes:
        clip_vcs = []
        for clip_id, frames_data in clip_frames.items():
            vc = compute_mVC_for_clip(
                frames_data["preds"],
                frames_data["gts"],
                n=window_n,
                background_label=0,
                skip_empty_windows=True,
            )
            if vc is not None:
                clip_vcs.append(vc)
        
        if clip_vcs:
            mvc_metrics[f"mVC_{window_n}"] = np.mean(clip_vcs)
        else:
            mvc_metrics[f"mVC_{window_n}"] = 0.0
    
    # Print detailed report
    print("\n" + "=" * 50)
    print(f"{'Class ID':<10} | {'IoU':<10} | {'Dice':<10}")
    print("-" * 50)
    for c in classes_to_eval:
        print(f"Class {c:<4} | {final_per_class_iou[c]:.4f}     | {final_per_class_dice[c]:.4f}")
    print("-" * 50)
    print(f"{'OVERALL':<10} | {mIoU:.4f}     | {mDice:.4f}")
    print(f"{'mAP':<10} | {AP_total:.4f}")
    print("-" * 50)
    print("Temporal Consistency (mVC - Mean across clips):")
    for window_n in window_sizes:
        print(f"  mVC_{window_n:<2} | {mvc_metrics[f'mVC_{window_n}']:.4f}")
    print("=" * 50 + "\n")
    
    return {
        "mIoU": mIoU,
        "Dice": mDice,
        "AP": AP_total,
        "AP50": AP_50,
        "AP75": AP_75,
        "per_class_iou": final_per_class_iou,
        "per_class_dice": final_per_class_dice,
        "mVC_8": mvc_metrics["mVC_8"],
        "mVC_12": mvc_metrics["mVC_12"],
        "mVC_24": mvc_metrics["mVC_24"],
    }