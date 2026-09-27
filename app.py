import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

from gradcam import make_gradcam_heatmap, overlay_heatmap, get_hotspot_regions, draw_hotspot_boxes
from clinical_report import generate_clinical_report

# Page config
st.set_page_config(
    page_title="🧠 Brain Tumor AI Detector",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.result-box {
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.healthy-box {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border-left: 5px solid #28a745;
}
.tumor-box {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border-left: 5px solid #dc3545;
}
.report-box {
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    background: #f8f9fa;
    border-left: 5px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
TARGET_SIZE = 128  # must match the model's training input size


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('brain_tumor_model.h5')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_image(image, target_size=TARGET_SIZE):
    """Returns (model_input_batch, display_rgb_uint8_image)."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    resized = cv2.resize(image, (target_size, target_size))
    normalized = resized.astype(np.float32) / 255.0
    batch = np.expand_dims(normalized, axis=0)
    return batch, resized  # resized is uint8-range but still float; cast below


def to_display_uint8(resized_float_image):
    return np.clip(resized_float_image, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">🧠 Brain Tumor AI Detector</h1>', unsafe_allow_html=True)
    st.markdown("### Upload an MRI scan for AI-powered analysis with visual explainability")

    model = load_model()
    if model is None:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Info")
        st.info("""
        - **Architecture**: Custom CNN
        - **Input Size**: 128×128 pixels
        - **Classes**: 4 tumor types
        - **Framework**: TensorFlow
        - **Explainability**: Grad-CAM
        """)

        st.markdown("## 🎛️ Explainability Settings")
        hotspot_threshold = st.slider(
            "Hotspot sensitivity", min_value=0.3, max_value=0.8, value=0.5, step=0.05,
            help="Lower = more (and larger) regions flagged as hotspots. "
                 "Higher = only the strongest activation kept."
        )
        overlay_alpha = st.slider(
            "Heatmap overlay strength", min_value=0.2, max_value=0.7, value=0.45, step=0.05
        )

        st.markdown("## ⚠️ Disclaimer")
        st.warning(
            "This tool is for educational purposes only. It is not a "
            "medical device and does not provide a diagnosis. Always "
            "consult a qualified radiologist or physician."
        )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an MRI image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear MRI brain scan image"
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(pil_image, caption="MRI Brain Scan", use_column_width=True)

        with col2:
            st.subheader("🤖 AI Analysis")
            if st.button("🔍 Analyze with AI", type="primary"):
                with st.spinner("🧠 AI is analyzing..."):
                    try:
                        img_batch, resized_display = preprocess_image(pil_image)
                        display_uint8 = to_display_uint8(resized_display)

                        # --- Prediction ---
                        predictions = model.predict(img_batch, verbose=0)[0]
                        predicted_idx = int(np.argmax(predictions))
                        predicted_class = CLASSES[predicted_idx]
                        confidence = float(predictions[predicted_idx])

                        # --- Result banner ---
                        if predicted_class == "No Tumor":
                            st.markdown(f"""
                            <div class="result-box healthy-box">
                                <h3>✅ Prediction: {predicted_class}</h3>
                                <h4>🎯 Confidence: {confidence:.1%}</h4>
                                <p>No signs of tumor detected.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-box tumor-box">
                                <h3>⚠️ Prediction: {predicted_class}</h3>
                                <h4>🎯 Confidence: {confidence:.1%}</h4>
                                <p>Signs of {predicted_class.lower()} detected.</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # --- Probability bars ---
                        st.subheader("📊 All Probabilities")
                        for class_name, prob in zip(CLASSES, predictions):
                            st.progress(float(prob), text=f"{class_name}: {prob:.1%}")

                        # --- Grad-CAM explainability ---
                        heatmap, _, _ = make_gradcam_heatmap(img_batch, model, pred_index=predicted_idx)
                        overlaid, heatmap_resized = overlay_heatmap(display_uint8, heatmap, alpha=overlay_alpha)
                        regions = get_hotspot_regions(heatmap_resized, threshold=hotspot_threshold)
                        annotated = draw_hotspot_boxes(overlaid, regions)

                        st.session_state["last_result"] = {
                            "predicted_class": predicted_class,
                            "confidence": confidence,
                            "predictions": predictions,
                            "annotated": annotated,
                            "regions": regions,
                        }

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        # --- Explainability + Report section (full width, below the columns) ---
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]

            st.markdown("---")
            st.subheader("🔥 Where the AI Looked (Grad-CAM Heatmap)")
            hm_col1, hm_col2 = st.columns([1, 1])
            with hm_col1:
                st.image(result["annotated"], caption="Model attention heatmap + hotspot regions",
                          use_column_width=True)
            with hm_col2:
                st.markdown(
                    "**How to read this:** warmer colors (red/yellow) mark pixels "
                    "that pushed the model toward its prediction; cooler colors "
                    "(blue) had little influence. Numbered boxes mark the "
                    "strongest, spatially distinct hotspots."
                )
                if result["regions"]:
                    st.markdown(f"**{len(result['regions'])} hotspot(s) detected.**")
                else:
                    st.markdown("**No spatially concentrated hotspot detected** "
                                "at the current sensitivity — try lowering the "
                                "slider in the sidebar.")

            st.markdown("---")
            st.subheader("📋 Clinical-Style Explanation")
            report_md = generate_clinical_report(
                predicted_class=result["predicted_class"],
                confidence=result["confidence"],
                all_probs=result["predictions"],
                class_names=CLASSES,
                regions=result["regions"],
            )
            # Rendered as plain markdown (not injected into a raw HTML div) so
            # the ### headers and ** bold text actually render instead of
            # showing up as literal characters.
            with st.container(border=True):
                st.markdown(report_md)

            st.download_button(
                "⬇️ Download Report (Markdown)",
                data=report_md,
                file_name=f"brain_scan_report_{result['predicted_class'].lower()}.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()