"""
LeafGuard AI — Gradio app for Hugging Face Spaces (free tier).
Reuses model_loader.py and gradcam.py from the FastAPI backend.
"""

import os
from io import BytesIO
import base64

import gradio as gr
from PIL import Image
import torch

from model_loader import ModelManager
from inference import predict
from preprocessing import preprocess_image


# ---------------------------------------------------------------------------
# Lazy Grad-CAM imports (avoid hard-fail at startup if heavy deps missing)
# ---------------------------------------------------------------------------
def _load_gradcam():
    try:
        from gradcam import generate_gradcam, overlay_heatmap
        return generate_gradcam, overlay_heatmap
    except Exception as e:
        print(f"Grad-CAM import failed: {e}")
        return None, None


generate_gradcam, overlay_heatmap = _load_gradcam()


# ---------------------------------------------------------------------------
# Model manager (loads both checkpoints at Space startup)
# ---------------------------------------------------------------------------
checkpoint_dir = os.getenv("CHECKPOINT_DIR", "runs/models")
model_manager = ModelManager(checkpoint_dir)
load_status = model_manager.load_models()

print(f"Model loading status: {load_status}")


def format_confidence(val: float) -> str:
    return f"{val * 100:.1f}%"


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def analyze_leaf(image, model_choice):
    """
    Gradio entry point for single-model prediction or compare-both.
    model_choice: 'compare', 'cbam', or 'baseline'
    """
    if image is None:
        return "Please upload a leaf image first.", None, "Please upload a leaf image first."

    results_md = ""
    error = None

    try:
        if model_choice == "compare":
            baseline_result = predict(
                model_manager.get_model("baseline"),
                image,
                model_manager.get_classes(),
                model_manager.device,
            )
            cbam_result = predict(
                model_manager.get_model("cbam"),
                image,
                model_manager.get_classes(),
                model_manager.device,
            )
            agree = baseline_result["predicted_class"] == cbam_result["predicted_class"]
            results_md = f"""### Baseline MobileNetV3-Small
**Prediction:** {baseline_result['predicted_class']}  
**Confidence:** {format_confidence(baseline_result['confidence'])}

### MobileNetV3-Small + CBAM
**Prediction:** {cbam_result['predicted_class']}  
**Confidence:** {format_confidence(cbam_result['confidence'])}

**Agreement:** {'Both models agree ✅' if agree else 'Models disagree ⚠️'}
"""
        else:
            model_name = model_choice  # 'cbam' or 'baseline'
            result = predict(
                model_manager.get_model(model_name),
                image,
                model_manager.get_classes(),
                model_manager.device,
            )
            results_md = f"""### {model_name.upper()}
**Prediction:** {result['predicted_class']}  
**Confidence:** {format_confidence(result['confidence'])}
"""
    except Exception as e:
        error = f"Inference failed: {str(e)}"

    return error or results_md, None, error or "Analysis complete."


def generate_explanation(image, model_choice):
    """Grad-CAM heatmap overlay."""
    if image is None:
        return None, "Please upload a leaf image first."

    if generate_gradcam is None or overlay_heatmap is None:
        return None, "Grad-CAM is not available on this Space."

    try:
        model_name = "cbam" if model_choice == "compare" else model_choice
        model_obj = model_manager.get_model(model_name)

        prediction = predict(
            model_obj,
            image,
            model_manager.get_classes(),
            model_manager.device,
        )

        heatmap = generate_gradcam(
            model_obj,
            image,
            prediction["predicted_label"],
            model_manager.device,
        )

        if heatmap is None:
            return None, "Grad-CAM generation returned no heatmap."

        overlay = overlay_heatmap(image, heatmap)
        return overlay, "Grad-CAM explanation ready."

    except Exception as e:
        return None, f"Grad-CAM failed: {str(e)}"


# ---------------------------------------------------------------------------
# Gradio Blocks UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="LeafGuard AI — Crop Leaf Disease Identification") as demo:
    gr.Markdown("""# 🍃 LeafGuard AI
    Multi-class crop leaf disease identification with **CBAM + MobileNetV3-Small** and Grad-CAM explainability.

    Upload a leaf image and choose which model(s) to run.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload leaf image")
            model_choice = gr.Dropdown(
                choices=[
                    ("Compare both (CBAM + Baseline)", "compare"),
                    ("CBAM only", "cbam"),
                    ("Baseline only", "baseline"),
                ],
                value="compare",
                label="Model",
            )
            analyze_btn = gr.Button("Analyze leaf", variant="primary")
            explain_btn = gr.Button("Show Grad-CAM explanation")
        with gr.Column(scale=1):
            result_md = gr.Markdown()
            gradcam_output = gr.Image(label="Grad-CAM heatmap")
            status_msg = gr.Markdown()

    analyze_btn.click(
        fn=analyze_leaf,
        inputs=[image_input, model_choice],
        outputs=[result_md, gradcam_output, status_msg],
    )

    explain_btn.click(
        fn=generate_explanation,
        inputs=[image_input, model_choice],
        outputs=[gradcam_output, status_msg],
    )

    gr.Markdown("""
    ---
    **Models:** Baseline MobileNetV3-Small vs MobileNetV3-Small + CBAM (Convolutional Block Attention Module)

    **Deployed on:** Hugging Face Spaces (free tier)

    **GitHub:** https://github.com/iamgkkj/LeafGuard
    """)

# Allow both `python app.py` (local) and HF Spaces launch
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
