import argparse
import os
import random
import shutil
import sys
from glob import glob
from pathlib import Path

try:
    import albumentations as A
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from albumentations.pytorch import ToTensorV2
    from sklearn.model_selection import train_test_split
    from skimage import io
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
except ImportError as e:
    print(f"\nMissing required dependency: {e.name}")
    print(
        "Please install prerequisites: "
        "pip install numpy nibabel tqdm matplotlib scikit-image "
        "torch torchvision scikit-learn opencv-python albumentations"
    )
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Hybrid GAT U-Net for MSD pancreas tumor segmentation."
    )
    parser.add_argument(
        "--dataset-folder",
        default=os.environ.get("PANCREAS_DATASET_FOLDER"),
        help="Path to the extracted Task07_Pancreas dataset root.",
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get(
            "PANCREAS_PROJECT_ROOT", str(Path(__file__).resolve().parent)
        ),
        help="Directory where outputs and checkpoints will be stored.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("PANCREAS_OUT_DIR"),
        help="Directory for preprocessed PNG slices.",
    )
    parser.add_argument(
        "--model-save-path",
        default=os.environ.get("PANCREAS_MODEL_SAVE_PATH"),
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("PANCREAS_NUM_EPOCHS", 60)),
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("PANCREAS_BATCH_SIZE", 4)),
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(
            os.environ.get("PANCREAS_WORKERS", max(0, (os.cpu_count() or 1) - 1))
        ),
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=int(os.environ.get("PANCREAS_IMAGE_SIZE", 256)),
        help="Input image size after resizing.",
    )
    parser.add_argument(
        "--healthy-slice-prob",
        type=float,
        default=float(os.environ.get("PANCREAS_HEALTHY_SLICE_PROB", 0.02)),
        help="Probability of keeping a slice without tumor pixels.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=float(os.environ.get("PANCREAS_VAL_SIZE", 0.2)),
        help="Fraction of case IDs used for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("PANCREAS_SEED", 42)),
        help="Random seed.",
    )
    parser.add_argument(
        "--force-preprocessing",
        action="store_true",
        help="Delete existing PNG slices and regenerate.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=os.environ.get("PANCREAS_DEVICE", "auto"),
        help="Training device.",
    )

    args = parser.parse_args()

    if not args.dataset_folder:
        parser.error(
            "--dataset-folder is required unless PANCREAS_DATASET_FOLDER is set."
        )

    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else project_root / "pancreas_preprocessed"
    )
    model_save_path = (
        Path(args.model_save_path).expanduser().resolve()
        if args.model_save_path
        else project_root / "best_gat_unet.pth"
    )

    return {
        "dataset_folder": Path(args.dataset_folder).expanduser().resolve(),
        "project_root": project_root,
        "out_dir": out_dir,
        "model_save_path": model_save_path,
        "visual_save_path": project_root / "visual_result_sample.png",
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "image_size": args.image_size,
        "healthy_slice_prob": args.healthy_slice_prob,
        "val_size": args.val_size,
        "seed": args.seed,
        "force_preprocessing": args.force_preprocessing,
        "device": args.device,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs(out_dir):
    images_out = out_dir / "images"
    masks_out = out_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    return images_out, masks_out


def reset_preprocessing_dirs(images_out, masks_out):
    for directory in (images_out, masks_out):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def window_and_normalize(image, wl=40, ww=400):
    tmin = wl - ww // 2
    tmax = wl + ww // 2
    image = np.clip(image, tmin, tmax)
    image = (image - tmin) / ww
    return image


def case_id_from_slice_path(path):
    stem = path.stem
    return stem.rsplit("_slice", 1)[0]


def run_preprocessing(config, images_out, masks_out):
    print("\n--- Starting Phase 1: Local Preprocessing ---")

    dataset_folder = config["dataset_folder"]
    images_tr_dir = dataset_folder / "imagesTr"
    labels_tr_dir = dataset_folder / "labelsTr"

    if not images_tr_dir.is_dir() or not labels_tr_dir.is_dir():
        print(f"Error: Could not find valid MSD Pancreas structure at {dataset_folder}")
        sys.exit(1)

    image_nii_files = sorted(glob(str(images_tr_dir / "*.nii.gz")))
    if not image_nii_files:
        print(f"Error: No training volumes found in {images_tr_dir}")
        sys.exit(1)

    existing_image_count = len(list(images_out.glob("*.png")))
    existing_mask_count = len(list(masks_out.glob("*.png")))

    if config["force_preprocessing"]:
        print("Force preprocessing enabled. Clearing existing PNG slices.")
        reset_preprocessing_dirs(images_out, masks_out)
    elif existing_image_count or existing_mask_count:
        if existing_image_count != existing_mask_count:
            print("Error: Existing preprocessing output is incomplete.")
            sys.exit(1)
        print(
            f"Found {existing_image_count} preprocessed image/mask pairs. "
            "Skipping preprocessing."
        )
        return

    print(f"Found {len(image_nii_files)} training volumes to process.")

    for img_path in tqdm(image_nii_files, desc="Processing Volumes"):
        case_id = Path(img_path).name.replace(".nii.gz", "")
        label_path = labels_tr_dir / f"{case_id}.nii.gz"

        if not label_path.exists():
            continue

        try:
            img = nib.load(img_path).get_fdata()
            lbl = nib.load(str(label_path)).get_fdata().astype(np.uint8)
        except Exception as e:
            print(f"Failed to load case {case_id}: {e}")
            continue

        img = window_and_normalize(img, wl=40, ww=400)

        for slice_idx in range(img.shape[2]):
            tumor_mask_slice = (lbl[:, :, slice_idx] == 2).astype(np.uint8)

            if tumor_mask_slice.sum() == 0 and random.random() > config["healthy_slice_prob"]:
                continue

            img_slice = img[:, :, slice_idx]
            img_fname = images_out / f"{case_id}_slice{slice_idx:03d}.png"
            mask_fname = masks_out / f"{case_id}_slice{slice_idx:03d}.png"

            io.imsave(str(img_fname), (img_slice * 255).astype(np.uint8), check_contrast=False)
            io.imsave(str(mask_fname), (tumor_mask_slice * 255).astype(np.uint8), check_contrast=False)

    print(f"Successfully generated PNG slices in {config['out_dir']}")


def collect_image_mask_pairs(images_out, masks_out):
    image_paths = sorted(images_out.glob("*.png"))
    mask_paths = sorted(masks_out.glob("*.png"))

    image_lookup = {path.name: path for path in image_paths}
    mask_lookup = {path.name: path for path in mask_paths}
    common_names = sorted(set(image_lookup) & set(mask_lookup))

    if not common_names:
        print("Error: No matching image/mask PNG pairs found.")
        sys.exit(1)

    return [(image_lookup[name], mask_lookup[name]) for name in common_names]


def split_pairs_by_case(pairs, val_size, seed):
    case_to_pairs = {}
    for image_path, mask_path in pairs:
        case_to_pairs.setdefault(case_id_from_slice_path(image_path), []).append(
            (image_path, mask_path)
        )

    case_ids = sorted(case_to_pairs)
    train_case_ids, val_case_ids = train_test_split(
        case_ids, test_size=val_size, random_state=seed
    )

    train_pairs = [pair for case_id in train_case_ids for pair in case_to_pairs[case_id]]
    val_pairs = [pair for case_id in val_case_ids for pair in case_to_pairs[case_id]]

    return train_pairs, val_pairs, train_case_ids, val_case_ids


# ==========================================
# DATASET
# ==========================================
class PancreasDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Failed to read pair at idx {idx}")

        mask = np.where(mask > 127, 1.0, (mask == 2).astype(np.float32)).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"].float()
            mask = augmented["mask"].float()
        else:
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            mask = torch.from_numpy(mask).float()

        if image.ndim == 2:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask


# ==========================================
# ORIGINAL ARCHITECTURE (exactly as trained before)
# ==========================================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.W(x)
        a1 = self.a.weight[0, :h.size(-1)].view(-1, 1)
        a2 = self.a.weight[0, h.size(-1):].view(-1, 1)
        score1 = torch.matmul(h, a1)
        score2 = torch.matmul(h, a2)
        e = self.leakyrelu(score1 + score2.transpose(1, 2))
        attention = F.softmax(e, dim=-1)
        h_prime = torch.bmm(attention, h)
        return F.elu(h_prime)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class HybridGATUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.b_conv = ConvBlock(256, 512)
        self.gat = GraphAttentionLayer(512, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d1 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d2 = ConvBlock(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d3 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool1(x1))
        x3 = self.e3(self.pool2(x2))

        b = self.b_conv(self.pool3(x3))
        B, C, H, W = b.size()
        nodes = b.view(B, C, -1).permute(0, 2, 1)
        gat_nodes = self.gat(nodes)
        b = gat_nodes.permute(0, 2, 1).view(B, C, H, W)

        d1 = self.up1(b)
        d1 = torch.cat([d1, x3], dim=1)
        d1 = self.d1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.d2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, x1], dim=1)
        d3 = self.d3(d3)

        return self.out(d3)


# ==========================================
# ORIGINAL LOSS (pos_weight=100, same as before)
# ==========================================
class TumorFocusLoss(nn.Module):
    def __init__(self, pos_weight=100.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets, smooth=1.0):
        probs = torch.sigmoid(inputs)
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)

        bce = F.binary_cross_entropy(probs_flat, targets_flat, reduction="none")
        weight_vector = targets_flat * self.pos_weight + (1 - targets_flat)
        weighted_bce = (bce * weight_vector).mean()

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + smooth) / (
            probs_flat.sum() + targets_flat.sum() + smooth
        )

        return weighted_bce + (1.0 - dice)


# ==========================================
# ORIGINAL TRANSFORMS (same as before)
# ==========================================
def create_transforms(image_size):
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])
    return train_transform, val_transform


def resolve_device(requested_device):
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            sys.exit("Error: CUDA not available.")
        return torch.device("cuda")
    if requested_device == "mps":
        if not mps_available:
            sys.exit("Error: MPS not available.")
        return torch.device("mps")
    if requested_device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def save_visual_sample(image_tensor, mask_tensor, pred_prob_tensor, save_path):
    display_image = image_tensor.cpu().numpy().squeeze()
    display_image = np.clip((display_image * 0.5) + 0.5, 0.0, 1.0)
    display_mask = mask_tensor.cpu().numpy().squeeze()
    display_pred = pred_prob_tensor.cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display_image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("CT Scan")
    axes[0].axis("off")
    axes[1].imshow(display_mask, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    axes[2].imshow(display_pred, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("GAT Prediction (Heatmap)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\nSaved visual sample result locally to: {save_path}")


# ==========================================
# RESEARCH ANALYTICS (for paper)
# ==========================================
def save_research_analytics(all_dice_scores, all_iou_scores, all_gt_labels, all_pred_probs,
                            tp, tn, fp, fn, train_losses, val_losses, analytics_dir):
    analytics_dir.mkdir(parents=True, exist_ok=True)

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = ["No Tumor", "Tumor"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
    ax.set_title("Confusion Matrix (Slice-Level)", fontsize=14, fontweight="bold")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=20,
                    fontweight="bold", color=color)
    plt.tight_layout()
    plt.savefig(analytics_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if all_dice_scores:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_dice_scores, bins=30, color="#2196F3", edgecolor="black", alpha=0.8)
        mean_d = np.mean(all_dice_scores)
        median_d = np.median(all_dice_scores)
        ax.axvline(mean_d, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_d:.3f}")
        ax.axvline(median_d, color="orange", linestyle="--", linewidth=2, label=f"Median: {median_d:.3f}")
        ax.set_xlabel("Dice Score", fontsize=13)
        ax.set_ylabel("Number of Slices", fontsize=13)
        ax.set_title("Dice Score Distribution (Tumor Slices)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(analytics_dir / "dice_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if all_iou_scores:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_iou_scores, bins=30, color="#4CAF50", edgecolor="black", alpha=0.8)
        ax.axvline(np.mean(all_iou_scores), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {np.mean(all_iou_scores):.3f}")
        ax.set_xlabel("IoU Score", fontsize=13)
        ax.set_ylabel("Number of Slices", fontsize=13)
        ax.set_title("IoU Score Distribution (Tumor Slices)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(analytics_dir / "iou_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if train_losses and val_losses:
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs_range = range(1, len(train_losses) + 1)
        ax.plot(epochs_range, train_losses, label="Train Loss", color="#2196F3", linewidth=2)
        ax.plot(epochs_range, val_losses, label="Val Loss", color="#F44336", linewidth=2)
        best_epoch = int(np.argmin(val_losses)) + 1
        ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.5,
                   label=f"Best Epoch: {best_epoch}")
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel("Loss", fontsize=13)
        ax.set_title("Training & Validation Loss Curve", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(analytics_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if all_gt_labels and all_pred_probs:
        gt_arr = np.concatenate(all_gt_labels)
        pred_arr = np.concatenate(all_pred_probs)
        if gt_arr.sum() > 0 and (1 - gt_arr).sum() > 0:
            thresholds = np.linspace(0, 1, 50)
            precisions, recalls, fprs = [], [], []
            for t in thresholds:
                pred_t = (pred_arr >= t).astype(np.float32)
                tp_px = ((pred_t == 1) & (gt_arr == 1)).sum()
                fp_px = ((pred_t == 1) & (gt_arr == 0)).sum()
                fn_px = ((pred_t == 0) & (gt_arr == 1)).sum()
                tn_px = ((pred_t == 0) & (gt_arr == 0)).sum()
                precisions.append(tp_px / (tp_px + fp_px + 1e-8))
                recalls.append(tp_px / (tp_px + fn_px + 1e-8))
                fprs.append(fp_px / (fp_px + tn_px + 1e-8))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            ax1.plot(recalls, precisions, color="#9C27B0", linewidth=2)
            ax1.set(xlabel="Recall", ylabel="Precision", xlim=[0, 1], ylim=[0, 1])
            ax1.set_title("Precision-Recall Curve (Pixel-Level)", fontsize=14, fontweight="bold")
            ax1.grid(True, alpha=0.3)
            ax2.plot(fprs, recalls, color="#FF9800", linewidth=2)
            ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax2.set(xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=[0, 1], ylim=[0, 1])
            ax2.set_title("ROC Curve (Pixel-Level)", fontsize=14, fontweight="bold")
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(analytics_dir / "pr_roc_curves.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    total = tp + tn + fp + fn
    precision_det = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_det = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_det = 2 * precision_det * recall_det / (precision_det + recall_det + 1e-8)
    specificity_det = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy_det = (tp + tn) / total if total > 0 else 0.0
    mean_dice = np.mean(all_dice_scores) if all_dice_scores else 0.0
    mean_iou = np.mean(all_iou_scores) if all_iou_scores else 0.0
    std_dice = np.std(all_dice_scores) if all_dice_scores else 0.0

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    table_data = [
        ["Detection Accuracy", f"{accuracy_det:.2%}"],
        ["Sensitivity (Recall)", f"{recall_det:.2%}"],
        ["Specificity", f"{specificity_det:.2%}"],
        ["Precision", f"{precision_det:.2%}"],
        ["F1 Score", f"{f1_det:.2%}"],
        ["", ""],
        ["Mean Dice (±SD)", f"{mean_dice:.2%} ± {std_dice:.2%}"],
        ["Mean IoU", f"{mean_iou:.2%}"],
        ["Median Dice", f"{np.median(all_dice_scores):.2%}" if all_dice_scores else "N/A"],
        ["", ""],
        ["TP / TN / FP / FN", f"{tp} / {tn} / {fp} / {fn}"],
        ["Total Slices", f"{total}"],
        ["Tumor Slices", f"{len(all_dice_scores)}"],
    ]
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"],
                     cellLoc="center", loc="center", colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ECEFF1")
    ax.set_title("Research Paper Metrics Summary\nHybrid GAT-UNet on MSD Pancreas Task07",
                 fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(analytics_dir / "metrics_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n--- Research Analytics saved to: {analytics_dir} ---")
    for name in ["confusion_matrix", "dice_distribution", "iou_distribution",
                  "loss_curve", "pr_roc_curves", "metrics_summary_table"]:
        print(f"  - {name}.png")


# ==========================================
# EVALUATION
# ==========================================
def run_final_evaluation(model, loader, device, visual_save_path,
                         train_losses=None, val_losses=None, analytics_dir=None):
    print("\n--- Starting Phase 5: Final Evaluation ---")
    model.eval()

    total_dice = total_iou = total_pixel_acc = 0.0
    slices_with_tumors = total_slices = 0
    tp = tn = fp = fn = 0
    visual_saved = False

    all_dice_scores, all_iou_scores = [], []
    all_gt_labels, all_pred_probs = [], []
    pixel_sample_interval = 8

    eval_loop = tqdm(loader, desc="Master Evaluation")
    with torch.no_grad():
        for images, masks in eval_loop:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            preds_raw = model(images)
            preds_prob = torch.sigmoid(preds_raw)
            preds_bin = (preds_prob > 0.5).float()

            if not visual_saved:
                has_tumor_batch = masks.reshape(masks.size(0), -1).sum(dim=1) > 0
                if has_tumor_batch.any():
                    idx = torch.where(has_tumor_batch)[0][0]
                    save_visual_sample(images[idx], masks[idx], preds_prob[idx], visual_save_path)
                    visual_saved = True

            for j in range(images.size(0)):
                total_slices += 1
                pred_flat = preds_bin[j].reshape(-1)
                mask_flat = masks[j].reshape(-1)

                has_tumor_true = mask_flat.sum().item() > 0
                has_tumor_pred = pred_flat.sum().item() > 0

                if has_tumor_true and has_tumor_pred:
                    tp += 1
                elif not has_tumor_true and not has_tumor_pred:
                    tn += 1
                elif not has_tumor_true and has_tumor_pred:
                    fp += 1
                else:
                    fn += 1

                if has_tumor_true:
                    intersection = (pred_flat * mask_flat).sum()
                    dice = (2.0 * intersection) / (pred_flat.sum() + mask_flat.sum() + 1e-8)
                    iou = intersection / (pred_flat.sum() + mask_flat.sum() - intersection + 1e-8)
                    total_dice += dice.item()
                    total_iou += iou.item()
                    all_dice_scores.append(dice.item())
                    all_iou_scores.append(iou.item())
                    slices_with_tumors += 1

                correct_pixels = (pred_flat == mask_flat).sum().item()
                total_pixel_acc += correct_pixels / mask_flat.numel()

                gt_sampled = mask_flat[::pixel_sample_interval].cpu().numpy()
                pred_sampled = preds_prob[j].reshape(-1)[::pixel_sample_interval].cpu().numpy()
                all_gt_labels.append(gt_sampled)
                all_pred_probs.append(pred_sampled)

    final_dice = total_dice / slices_with_tumors if slices_with_tumors > 0 else 0.0
    final_iou = total_iou / slices_with_tumors if slices_with_tumors > 0 else 0.0
    detection_accuracy = (tp + tn) / total_slices if total_slices > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
    avg_pixel_acc = total_pixel_acc / total_slices if total_slices > 0 else 0.0

    if slices_with_tumors > 0:
        print(f"\n{'=' * 30}")
        print("--- FINAL PROJECT METRICS ---")
        print(f"{'=' * 30}")
        print(f"Tumor Slices Evaluated: {slices_with_tumors}")
        print(f"Average Dice Score: {final_dice * 100:.2f}%")
        print(f"Average IoU Score:  {final_iou * 100:.2f}%")
        print(f"{'=' * 30}")

    print(f"\n{'=' * 50}")
    print("              FINAL AI METRICS REPORT")
    print(f"{'=' * 50}")
    print("\n--- 1. DETECTION ACCURACY ---")
    print(f"Total Slices Tested : {total_slices}")
    print(f"True Positives (TP) : {tp}")
    print(f"True Negatives (TN) : {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print("-" * 50)
    print(f"Detection Accuracy  : {detection_accuracy * 100:.2f}%")
    print(f"Sensitivity (Recall): {sensitivity * 100:.2f}%")
    print(f"Specificity         : {specificity * 100:.2f}%")
    print(f"Precision           : {precision * 100:.2f}%")
    print(f"F1 Score            : {f1_score * 100:.2f}%")

    print("\n--- 2. LOCALIZATION ACCURACY ---")
    print(f"Tumor Slices Graded : {slices_with_tumors}")
    print(f"Average Dice Score  : {final_dice * 100:.2f}%")
    print(f"Dice Std Dev        : {np.std(all_dice_scores) * 100:.2f}%")
    print(f"Median Dice Score   : {np.median(all_dice_scores) * 100:.2f}%")
    print(f"Average IoU Score   : {final_iou * 100:.2f}%")

    print("\n--- 3. OVERALL PIXEL ACCURACY ---")
    print(f"Average Pixel Acc   : {avg_pixel_acc * 100:.2f}%")
    print("=" * 50)

    if analytics_dir:
        save_research_analytics(all_dice_scores, all_iou_scores, all_gt_labels,
                                all_pred_probs, tp, tn, fp, fn,
                                train_losses or [], val_losses or [], analytics_dir)

    print("Local Training and Evaluation Complete.")


# ==========================================
# MAIN
# ==========================================
def main():
    config = parse_args()
    set_seed(config["seed"])
    config["project_root"].mkdir(parents=True, exist_ok=True)
    config["out_dir"].mkdir(parents=True, exist_ok=True)
    images_out, masks_out = ensure_output_dirs(config["out_dir"])

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    run_preprocessing(config, images_out, masks_out)

    all_pairs = collect_image_mask_pairs(images_out, masks_out)
    print(f"\nPhase 2: Data splitting. Found {len(all_pairs)} generated PNG slices total.")

    train_pairs, val_pairs, train_case_ids, val_case_ids = split_pairs_by_case(
        all_pairs, config["val_size"], config["seed"]
    )
    print(f"Case-level split: {len(train_case_ids)} train cases, {len(val_case_ids)} val cases.")
    print(f"Slice totals: {len(train_pairs)} train, {len(val_pairs)} val.")

    train_imgs, train_masks = zip(*train_pairs)
    val_imgs, val_masks = zip(*val_pairs)

    train_transform, val_transform = create_transforms(config["image_size"])
    train_dataset = PancreasDataset(train_imgs, train_masks, transform=train_transform)
    val_dataset = PancreasDataset(val_imgs, val_masks, transform=val_transform)

    device = resolve_device(config["device"])
    print(f"\n--- Initializing Phase 3: Local Training on {device} ---")
    if device.type == "mps":
        print("Using Apple Metal Performance Shaders (MPS) for GPU training.")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["workers"], pin_memory=pin_memory,
                              persistent_workers=config["workers"] > 0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["workers"], pin_memory=pin_memory,
                            persistent_workers=config["workers"] > 0)

    model = HybridGATUNet().to(device)
    criterion = TumorFocusLoss(pos_weight=100.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []

    for epoch in range(config["num_epochs"]):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, leave=True, desc=f"Epoch [{epoch + 1}/{config['num_epochs']}] Train")
        for images, masks in loop:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.5f}")

        avg_train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, leave=False, desc="Validation"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_running += loss.item()

        avg_val_loss = val_running / max(1, len(val_loader))

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config["model_save_path"])
            print(f"-> Best model saved locally to: {config['model_save_path']}")

    if not config["model_save_path"].exists():
        print("Error: Model checkpoint was not saved during training.")
        sys.exit(1)

    model.load_state_dict(torch.load(config["model_save_path"], map_location=device))
    analytics_dir = config["project_root"] / "research_analytics"
    run_final_evaluation(model, val_loader, device, config["visual_save_path"],
                         train_losses=train_loss_history, val_losses=val_loss_history,
                         analytics_dir=analytics_dir)


if __name__ == "__main__":
    main()
