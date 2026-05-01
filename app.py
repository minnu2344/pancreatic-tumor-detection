"""
Pancreatic Tumor Detection & Localization — Radiologist Dashboard

A Streamlit web UI for uploading CT scan slices, running the Hybrid GAT U-Net
model, and viewing detection/localization results with overlays and metrics.
"""

import io
import os
import sys
from pathlib import Path

import albumentations as A
import base64
import cv2
import groq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import HybridGATUNet

PROJECT_ROOT = Path(__file__).resolve().parent / "run_pancreas_m3"
_model_in_project = PROJECT_ROOT / "best_gat_unet.pth"
_model_in_root = Path(__file__).resolve().parent / "best_gat_unet.pth"
MODEL_PATH = _model_in_project if _model_in_project.exists() else _model_in_root
IMAGE_SIZE = 256


# ── helpers ──────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@st.cache_resource
def load_model():
    device = get_device()
    model = HybridGATUNet().to(device)
    model.load_state_dict(
        torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
    )
    model.eval()
    return model, device


def preprocess(image_gray):
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    return transform(image=image_rgb)["image"].unsqueeze(0).float()


def run_inference(model, tensor, device):
    with torch.no_grad():
        logits = model(tensor.to(device))
        return torch.sigmoid(logits).squeeze().cpu().numpy()


def compute_metrics(gt_mask, pred_binary):
    g = gt_mask.flatten()
    p = pred_binary.flatten()
    inter = (g * p).sum()
    dice = (2.0 * inter) / (g.sum() + p.sum() + 1e-8)
    iou = inter / (g.sum() + p.sum() - inter + 1e-8)
    tp = ((g == 1) & (p == 1)).sum()
    tn = ((g == 0) & (p == 0)).sum()
    fp = ((g == 0) & (p == 1)).sum()
    fn = ((g == 1) & (p == 0)).sum()
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return {
        "dice": float(dice), "iou": float(iou),
        "accuracy": float(accuracy), "sensitivity": float(sensitivity),
        "specificity": float(specificity), "precision": float(precision),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "gt_pixels": int(g.sum()), "pred_pixels": int(p.sum()),
    }


def get_contours(mask):
    m = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


def draw_contours_on_ax(ax, contours, color, linewidth=2, label=None):
    from matplotlib.patches import Polygon
    for i, cnt in enumerate(contours):
        if len(cnt) < 3:
            continue
        pts = cnt.squeeze()
        if pts.ndim != 2:
            continue
        ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=color,
                             linewidth=linewidth, label=label if i == 0 else None))


def build_report_figure(ct, gt_mask, pred_binary, heatmap, title, metrics):
    gt_contours = get_contours(gt_mask)
    pred_contours = get_contours(pred_binary)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    axes[0, 0].imshow(ct, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("(a) Input CT Scan", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ct, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*ct.shape, 4))
    overlay[gt_mask > 0] = [1.0, 0.0, 0.0, 0.45]
    axes[0, 1].imshow(overlay)
    draw_contours_on_ax(axes[0, 1], gt_contours, "lime", 2.5, "Tumor Boundary")
    axes[0, 1].set_title("(b) Ground Truth Tumor on CT", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")
    if gt_contours:
        axes[0, 1].legend(loc="upper right", fontsize=8, framealpha=0.8)

    axes[1, 0].imshow(ct, cmap="gray", vmin=0, vmax=1)
    im = axes[1, 0].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1, 0].set_title("(c) Prediction Heatmap on CT", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
    cbar = plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label("Tumor Probability", fontsize=9)

    axes[1, 1].imshow(ct, cmap="gray", vmin=0, vmax=1)
    pred_ov = np.zeros((*ct.shape, 4))
    pred_ov[pred_binary > 0] = [0.0, 0.8, 1.0, 0.3]
    axes[1, 1].imshow(pred_ov)
    draw_contours_on_ax(axes[1, 1], gt_contours, "lime", 2.5, "Ground Truth")
    draw_contours_on_ax(axes[1, 1], pred_contours, "cyan", 2.5, "Prediction")
    axes[1, 1].set_title("(d) GT vs Prediction Comparison", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")
    if gt_contours or pred_contours:
        axes[1, 1].legend(loc="upper right", fontsize=9, framealpha=0.8)

    if metrics:
        axes[1, 1].text(
            0.02, 0.02,
            f"Dice: {metrics['dice']:.2%}\nIoU:  {metrics['iou']:.2%}\n"
            f"GT pixels: {metrics['gt_pixels']}\nPred pixels: {metrics['pred_pixels']}",
            transform=axes[1, 1].transAxes, fontsize=9, color="white",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.75),
        )

    plt.tight_layout()
    return fig


def build_inference_only_figure(ct, heatmap, pred_binary):
    pred_contours = get_contours(pred_binary)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Tumor Detection & Localization Result", fontsize=14, fontweight="bold")

    axes[0].imshow(ct, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Input CT Scan", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(ct, cmap="gray", vmin=0, vmax=1)
    im = axes[1].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Prediction Heatmap", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label("Probability", fontsize=9)

    axes[2].imshow(ct, cmap="gray", vmin=0, vmax=1)
    pred_ov = np.zeros((*ct.shape, 4))
    pred_ov[pred_binary > 0] = [0.0, 1.0, 0.5, 0.4]
    axes[2].imshow(pred_ov)
    draw_contours_on_ax(axes[2], pred_contours, "lime", 2.5)
    axes[2].set_title("Tumor Localization", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    tumor_pct = pred_binary.sum() / pred_binary.size * 100
    axes[2].text(
        0.02, 0.02,
        f"Tumor pixels: {int(pred_binary.sum())}\n"
        f"Coverage: {tumor_pct:.2f}%\n"
        f"Max confidence: {heatmap.max():.2%}",
        transform=axes[2].transAxes, fontsize=9, color="white",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.75),
    )

    plt.tight_layout()
    return fig


# ── AI Diagnosis (Groq + multimodal) ───────────────────────────────────────────
# Vision-capable model (see https://console.groq.com/docs/deprecations)
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"


def get_groq_client():
    api_key = ""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.session_state.get("groq_api_key", "")
    if api_key:
        return groq.Groq(api_key=api_key)
    return None


def _groq_key_is_preconfigured():
    try:
        return bool(st.secrets.get("GROQ_API_KEY"))
    except (KeyError, FileNotFoundError):
        pass
    return bool(os.environ.get("GROQ_API_KEY"))


def pil_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_diagnosis_prompt(
    tumor_detected, tumor_pixels, coverage_pct, max_confidence,
    threshold, metrics=None, bbox=None,
):
    parts = [
        "You are an expert radiologist AI assistant specializing in pancreatic imaging. ",
        "A CT scan slice of the abdomen has been analyzed by a Hybrid GAT U-Net deep-learning "
        "model trained on the Medical Segmentation Decathlon Task07 Pancreas dataset.\n\n",
        "## Model Analysis Results\n",
        f"- **Tumor Detected**: {'Yes' if tumor_detected else 'No'}\n",
        f"- **Tumor Pixels**: {tumor_pixels:,}\n",
        f"- **Tumor Coverage**: {coverage_pct:.2f}% of the slice\n",
        f"- **Maximum Confidence**: {max_confidence:.1%}\n",
        f"- **Detection Threshold**: {threshold}\n",
    ]

    if bbox:
        parts.append(
            f"- **Bounding Box**: X [{bbox['col_min']}-{bbox['col_max']}], "
            f"Y [{bbox['row_min']}-{bbox['row_max']}], "
            f"Size {bbox['width']}x{bbox['height']} px\n"
        )

    if metrics:
        parts.append("\n## Segmentation Accuracy (vs Ground Truth)\n")
        parts.append(f"- **Dice Score**: {metrics['dice']:.2%}\n")
        parts.append(f"- **IoU (Jaccard)**: {metrics['iou']:.2%}\n")
        parts.append(f"- **Sensitivity (Recall)**: {metrics['sensitivity']:.2%}\n")
        parts.append(f"- **Precision**: {metrics['precision']:.2%}\n")
        parts.append(f"- **Specificity**: {metrics['specificity']:.2%}\n")
        parts.append(f"- **Accuracy**: {metrics['accuracy']:.2%}\n")
        parts.append(f"- **True Positives**: {metrics['tp']}\n")
        parts.append(f"- **False Positives**: {metrics['fp']}\n")
        parts.append(f"- **False Negatives**: {metrics['fn']}\n")

    parts.append(
        "\n## Instructions\n"
        "Based on the CT scan image and the model's analysis above, provide:\n"
        "1. **Clinical Assessment**: Interpret the findings — is the tumor likely "
        "malignant or benign based on size, location, and morphology visible in the scan?\n"
        "2. **Possible Diagnoses**: List the most probable diagnoses (e.g., pancreatic "
        "ductal adenocarcinoma, neuroendocrine tumor, cystic neoplasm, etc.) with brief reasoning.\n"
        "3. **Severity Estimation**: Low / Moderate / High concern, and why.\n"
        "4. **Recommended Next Steps**: What additional imaging, labs, or procedures "
        "should be considered?\n"
        "5. **Model Confidence Commentary**: Comment on how confident the model appears "
        "and any caveats about false positives/negatives.\n\n"
        "**Important**: Clearly state that this is an AI-assisted preliminary analysis "
        "and NOT a substitute for professional medical diagnosis. "
        "Recommend consultation with a qualified radiologist/oncologist.\n"
        "Format your response with clear headings and bullet points."
    )

    return "".join(parts)


def get_ai_diagnosis(client, ct_image_pil, prompt):
    img_b64 = pil_to_base64(ct_image_pil)
    response = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            }
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return response.choices[0].message.content


# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pancreas Tumor Detection & Localization",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a73e8; margin-bottom: 0.2rem; }
    .sub-header  { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px; padding: 1rem 1.2rem;
        border-left: 4px solid #1a73e8; margin-bottom: 0.6rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #666; }
    .metric-card p  { margin: 0; font-size: 1.5rem; font-weight: 700; color: #1a73e8; }
    .detection-positive { border-left-color: #e53935; }
    .detection-positive p { color: #e53935; }
    .detection-negative { border-left-color: #43a047; }
    .detection-negative p { color: #43a047; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem; font-weight: 600;
        padding: 0.5rem 1.5rem; border-radius: 8px 8px 0 0;
    }
    .ai-diagnosis-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
        border: 1px solid #a8c7fa;
        border-radius: 12px; padding: 1.5rem;
        margin-top: 1rem; margin-bottom: 1rem;
    }
    .ai-diagnosis-box h4 { color: #1a56db; margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# ── header ───────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">Pancreatic Tumor Detection & Localization</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hybrid GAT U-Net &nbsp;|&nbsp; Upload CT scan slices for '
    'AI-powered tumor detection, segmentation, and localization</div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        f"Trained model not found at `{MODEL_PATH}`. "
        "Please train the model first by running `main.py`."
    )
    st.stop()

model, device = load_model()
st.sidebar.success(f"Model loaded on **{str(device).upper()}**")

# ── sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown("### How to Use")
st.sidebar.markdown(
    "1. Choose a mode below\n"
    "2. Upload a CT scan slice (`.png`)\n"
    "3. Optionally upload the ground truth mask\n"
    "4. View detection & localization results"
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.markdown(
    "- **Architecture**: Hybrid GAT U-Net\n"
    "- **Input**: 256 x 256 grayscale CT\n"
    "- **Task**: Pancreatic tumor segmentation\n"
    "- **Dataset**: MSD Task07 Pancreas"
)
st.sidebar.markdown("---")
threshold = st.sidebar.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05,
                               help="Pixels with probability above this are classified as tumor.")

st.sidebar.markdown("---")
st.sidebar.markdown("### AI Diagnosis (Groq)")

if _groq_key_is_preconfigured():
    st.sidebar.success("Groq API key loaded from secrets/environment")
else:
    groq_key_input = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        key="groq_key_widget",
        help="Get your free key at https://console.groq.com/keys",
    )
    if groq_key_input:
        st.session_state["groq_api_key"] = groq_key_input
    elif "groq_api_key" in st.session_state and not groq_key_input:
        del st.session_state["groq_api_key"]

groq_client = get_groq_client()
if not groq_client and not _groq_key_is_preconfigured():
    st.sidebar.warning("Enter a Groq API key to enable AI Diagnosis")


# ── tabs ─────────────────────────────────────────────────────────────────────

tab_single, tab_batch = st.tabs(["Single Scan Analysis", "Batch Analysis"])


# ── TAB 1: Single Scan ──────────────────────────────────────────────────────

with tab_single:
    col_upload, col_mask = st.columns(2)

    with col_upload:
        st.markdown("#### Upload CT Scan Slice")
        ct_file = st.file_uploader(
            "Select a preprocessed CT scan PNG",
            type=["png", "jpg", "jpeg"],
            key="ct_single",
            help="Upload from val_images/images/ folder",
        )

    with col_mask:
        st.markdown("#### Upload Ground Truth Mask (optional)")
        mask_file = st.file_uploader(
            "Select the corresponding mask PNG",
            type=["png", "jpg", "jpeg"],
            key="mask_single",
            help="Upload from val_images/masks/ folder for comparison",
        )

    if ct_file is not None:
        ct_bytes = np.frombuffer(ct_file.read(), np.uint8)
        ct_raw = cv2.imdecode(ct_bytes, cv2.IMREAD_GRAYSCALE)

        if ct_raw is None:
            st.error("Could not read the uploaded CT image.")
        else:
            with st.spinner("Running AI inference..."):
                input_tensor = preprocess(ct_raw)
                heatmap = run_inference(model, input_tensor, device)
                pred_binary = (heatmap > threshold).astype(np.float32)

            ct_resized = cv2.resize(ct_raw, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
            tumor_detected = pred_binary.sum() > 0
            tumor_pct = pred_binary.sum() / pred_binary.size * 100

            st.markdown("---")

            # Detection result banner
            if tumor_detected:
                st.markdown(
                    '<div class="metric-card detection-positive">'
                    '<h3>DETECTION RESULT</h3>'
                    '<p>TUMOR DETECTED</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="metric-card detection-negative">'
                    '<h3>DETECTION RESULT</h3>'
                    '<p>NO TUMOR DETECTED</p></div>',
                    unsafe_allow_html=True,
                )

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><h3>Tumor Pixels</h3>'
                    f'<p>{int(pred_binary.sum()):,}</p></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><h3>Coverage</h3>'
                    f'<p>{tumor_pct:.2f}%</p></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><h3>Max Confidence</h3>'
                    f'<p>{heatmap.max():.1%}</p></div>',
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f'<div class="metric-card"><h3>Threshold</h3>'
                    f'<p>{threshold}</p></div>',
                    unsafe_allow_html=True,
                )

            gt_mask = None
            metrics = None
            if mask_file is not None:
                mask_bytes = np.frombuffer(mask_file.read(), np.uint8)
                mask_raw = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
                if mask_raw is not None:
                    gt_mask = cv2.resize(mask_raw, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
                    gt_mask = (gt_mask > 0.5).astype(np.float32)
                    metrics = compute_metrics(gt_mask, pred_binary)

            if gt_mask is not None and metrics is not None:
                st.markdown("#### Segmentation Accuracy")
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    st.metric("Dice Score", f"{metrics['dice']:.2%}")
                with a2:
                    st.metric("IoU (Jaccard)", f"{metrics['iou']:.2%}")
                with a3:
                    st.metric("Sensitivity", f"{metrics['sensitivity']:.2%}")
                with a4:
                    st.metric("Precision", f"{metrics['precision']:.2%}")

                fig = build_report_figure(ct_resized, gt_mask, pred_binary, heatmap,
                                          f"Analysis: {ct_file.name}", metrics)
            else:
                fig = build_inference_only_figure(ct_resized, heatmap, pred_binary)

            report_img = fig_to_image(fig)
            plt.close(fig)
            st.image(report_img, use_container_width=True)

            # Localization info
            if tumor_detected:
                coords = np.where(pred_binary > 0)
                row_min, row_max = int(coords[0].min()), int(coords[0].max())
                col_min, col_max = int(coords[1].min()), int(coords[1].max())

                st.markdown("#### Tumor Localization (Bounding Box)")
                l1, l2, l3, l4 = st.columns(4)
                with l1:
                    st.metric("X range (px)", f"{col_min} - {col_max}")
                with l2:
                    st.metric("Y range (px)", f"{row_min} - {row_max}")
                with l3:
                    st.metric("Width (px)", f"{col_max - col_min + 1}")
                with l4:
                    st.metric("Height (px)", f"{row_max - row_min + 1}")

            # AI Diagnosis
            st.markdown("---")
            st.markdown("#### AI-Powered Diagnosis (Groq)")

            if not groq_client:
                st.info(
                    "Set `GROQ_API_KEY` in Streamlit secrets, a `.env` file, or the sidebar. "
                    "Get a free key at [console.groq.com/keys](https://console.groq.com/keys)."
                )
            else:
                bbox_info = None
                if tumor_detected:
                    coords = np.where(pred_binary > 0)
                    bbox_info = {
                        "row_min": int(coords[0].min()),
                        "row_max": int(coords[0].max()),
                        "col_min": int(coords[1].min()),
                        "col_max": int(coords[1].max()),
                        "width": int(coords[1].max() - coords[1].min() + 1),
                        "height": int(coords[0].max() - coords[0].min() + 1),
                    }

                if st.button("Get AI Diagnosis", type="primary", key="ai_diag_single"):
                    with st.spinner("Consulting Llama AI via Groq..."):
                        prompt = build_diagnosis_prompt(
                            tumor_detected=tumor_detected,
                            tumor_pixels=int(pred_binary.sum()),
                            coverage_pct=tumor_pct,
                            max_confidence=float(heatmap.max()),
                            threshold=threshold,
                            metrics=metrics,
                            bbox=bbox_info,
                        )
                        ct_gray = (ct_resized * 255).astype(np.uint8)
                        ct_rgb = cv2.cvtColor(ct_gray, cv2.COLOR_GRAY2RGB)
                        ct_pil = Image.fromarray(ct_rgb)
                        try:
                            diagnosis = get_ai_diagnosis(groq_client, ct_pil, prompt)
                            st.session_state["last_diagnosis"] = diagnosis
                        except Exception as exc:
                            st.error(f"Groq API error: {exc}")
                            st.session_state.pop("last_diagnosis", None)

                if "last_diagnosis" in st.session_state:
                    st.markdown(
                        '<div class="ai-diagnosis-box">',
                        unsafe_allow_html=True,
                    )
                    st.markdown(st.session_state["last_diagnosis"])
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")

            # Download button
            buf = io.BytesIO()
            report_img.save(buf, format="PNG")
            st.download_button(
                label="Download Report Image",
                data=buf.getvalue(),
                file_name=f"report_{ct_file.name}",
                mime="image/png",
            )


# ── TAB 2: Batch Analysis ───────────────────────────────────────────────────

with tab_batch:
    st.markdown("#### Upload Multiple CT Scan Slices")
    st.caption("Upload several CT slices to analyze them all at once. Ground truth masks are optional.")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        batch_ct_files = st.file_uploader(
            "CT Scan Slices",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="ct_batch",
        )
    with col_b2:
        batch_mask_files = st.file_uploader(
            "Ground Truth Masks (optional, same filenames)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="mask_batch",
        )

    if batch_ct_files:
        mask_lookup = {}
        if batch_mask_files:
            for mf in batch_mask_files:
                mask_lookup[mf.name] = mf

        results = []
        progress = st.progress(0, text="Analyzing scans...")

        for i, ct_f in enumerate(batch_ct_files):
            ct_bytes = np.frombuffer(ct_f.read(), np.uint8)
            ct_raw = cv2.imdecode(ct_bytes, cv2.IMREAD_GRAYSCALE)
            if ct_raw is None:
                continue

            input_tensor = preprocess(ct_raw)
            heatmap = run_inference(model, input_tensor, device)
            pred_binary = (heatmap > threshold).astype(np.float32)

            ct_resized = cv2.resize(ct_raw, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
            tumor_detected = pred_binary.sum() > 0
            max_conf = float(heatmap.max())

            metrics = None
            if ct_f.name in mask_lookup:
                mf = mask_lookup[ct_f.name]
                mf.seek(0)
                mb = np.frombuffer(mf.read(), np.uint8)
                mr = cv2.imdecode(mb, cv2.IMREAD_GRAYSCALE)
                if mr is not None:
                    gt = cv2.resize(mr, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
                    gt = (gt > 0.5).astype(np.float32)
                    metrics = compute_metrics(gt, pred_binary)

            results.append({
                "name": ct_f.name,
                "detected": tumor_detected,
                "pixels": int(pred_binary.sum()),
                "confidence": max_conf,
                "dice": metrics["dice"] if metrics else None,
                "iou": metrics["iou"] if metrics else None,
                "sensitivity": metrics["sensitivity"] if metrics else None,
            })

            progress.progress((i + 1) / len(batch_ct_files),
                              text=f"Analyzing {i + 1}/{len(batch_ct_files)}...")

        progress.empty()

        detected_count = sum(1 for r in results if r["detected"])
        total = len(results)

        st.markdown("---")
        st.markdown("#### Batch Results Summary")

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Total Scans", total)
        with s2:
            st.metric("Tumors Detected", f"{detected_count}/{total}")
        with s3:
            dices = [r["dice"] for r in results if r["dice"] is not None]
            if dices:
                st.metric("Mean Dice Score", f"{np.mean(dices):.2%}")
            else:
                st.metric("Mean Dice Score", "N/A (no masks)")

        header_cols = st.columns([3, 2, 2, 2, 2, 2])
        header_cols[0].markdown("**Filename**")
        header_cols[1].markdown("**Detection**")
        header_cols[2].markdown("**Tumor Pixels**")
        header_cols[3].markdown("**Max Conf.**")
        header_cols[4].markdown("**Dice**")
        header_cols[5].markdown("**IoU**")

        for r in results:
            cols = st.columns([3, 2, 2, 2, 2, 2])
            cols[0].text(r["name"])
            if r["detected"]:
                cols[1].markdown(":red[TUMOR]")
            else:
                cols[1].markdown(":green[CLEAR]")
            cols[2].text(f"{r['pixels']:,}")
            cols[3].text(f"{r['confidence']:.1%}")
            cols[4].text(f"{r['dice']:.2%}" if r["dice"] is not None else "--")
            cols[5].text(f"{r['iou']:.2%}" if r["iou"] is not None else "--")

        # Batch AI Diagnosis Summary
        st.markdown("---")
        st.markdown("#### AI-Powered Batch Summary (Groq)")

        if not groq_client:
            st.info(
                "Set `GROQ_API_KEY` in Streamlit secrets, a `.env` file, or the sidebar. "
                "Get a free key at [console.groq.com/keys](https://console.groq.com/keys)."
            )
        else:
            if st.button("Get AI Batch Summary", type="primary", key="ai_diag_batch"):
                with st.spinner("Consulting Llama AI for batch summary..."):
                    summary_lines = [
                        "You are an expert radiologist AI assistant. A batch of CT scan slices "
                        "was analyzed by a Hybrid GAT U-Net model trained on the MSD Task07 "
                        "Pancreas dataset. Here is the summary:\n\n",
                        f"- **Total Scans**: {total}\n",
                        f"- **Tumors Detected**: {detected_count}/{total}\n",
                    ]
                    if dices:
                        summary_lines.append(
                            f"- **Mean Dice Score**: {np.mean(dices):.2%}\n"
                        )

                    summary_lines.append("\n## Per-Scan Results\n")
                    for r in results:
                        line = (
                            f"- **{r['name']}**: "
                            f"{'TUMOR' if r['detected'] else 'CLEAR'}, "
                            f"{r['pixels']:,} tumor pixels, "
                            f"confidence {r['confidence']:.1%}"
                        )
                        if r["dice"] is not None:
                            line += f", Dice {r['dice']:.2%}, IoU {r['iou']:.2%}"
                        summary_lines.append(line + "\n")

                    summary_lines.append(
                        "\n## Instructions\n"
                        "Based on this batch analysis:\n"
                        "1. Provide an overall clinical assessment of the batch.\n"
                        "2. Flag which scans are most concerning and why.\n"
                        "3. Suggest priority ordering for radiologist review.\n"
                        "4. Note any patterns across slices (e.g., consistent "
                        "tumor location suggesting a single mass vs. scattered findings).\n"
                        "5. Recommend next steps for the patient.\n\n"
                        "**Important**: State this is AI-assisted analysis, not a "
                        "substitute for professional medical diagnosis.\n"
                        "Format with clear headings and bullet points."
                    )

                    batch_prompt = "".join(summary_lines)
                    try:
                        batch_response = groq_client.chat.completions.create(
                            model=GROQ_TEXT_MODEL,
                            messages=[{"role": "user", "content": batch_prompt}],
                            temperature=0.3,
                            max_tokens=2048,
                        )
                        st.session_state["batch_diagnosis"] = batch_response.choices[0].message.content
                    except Exception as exc:
                        st.error(f"Groq API error: {exc}")
                        st.session_state.pop("batch_diagnosis", None)

            if "batch_diagnosis" in st.session_state:
                st.markdown(
                    '<div class="ai-diagnosis-box">',
                    unsafe_allow_html=True,
                )
                st.markdown(st.session_state["batch_diagnosis"])
                st.markdown("</div>", unsafe_allow_html=True)


# ── footer ───────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Hybrid GAT U-Net for Pancreatic Tumor Detection & Localization<br>"
    "MSD Task07 Pancreas Dataset</small>",
    unsafe_allow_html=True,
)
