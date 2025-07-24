import sys
import os

# Project root folder (ek level upar, jahan 'src' folder hai)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')

# Agar 'src' path sys.path me nahi hai, toh add kar do
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import io

# Ab direct src ke andar modules import karo
from model_transfer_learning import build_model as build_resnet
from model_custom_cnn import build_model as build_custom
from preprocessing import get_class_labels

# ---------- CONFIG ----------
DATA_DIR = r"E:\Brain Tumor MRI Image Classification\Data\train\train"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LOAD MODEL ----------
@st.cache_resource(show_spinner=False)
def load_model(model_type):
    class_names = get_class_labels(DATA_DIR)
    num_classes = len(class_names)

    if model_type == "ResNet50":
        model = build_resnet("resnet50", num_classes)
        model_path = "outputs/models/best_model.pth"
    else:
        model = build_custom(num_classes)
        model_path = "outputs/models/custom_model.pth"

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, class_names

# ---------- GRAD-CAM ----------
def apply_grad_cam(model, image_tensor, target_layer_name="layer4"):
    grad = None
    activation = None

    def save_grad(module, input, output):
        nonlocal grad
        grad = output[0].detach()

    def save_activation(module, input, output):
        nonlocal activation
        activation = output.detach()

    # Register hooks on the target layer
    layer_found = False
    for name, module in model.named_modules():
        if name == target_layer_name or name.lower().startswith("conv"):  # fallback for custom CNN
            layer_found = True
            module.register_forward_hook(save_activation)
            module.register_full_backward_hook(lambda m, gi, go: save_grad(m, gi, go))
            break

    if not layer_found:
        raise ValueError(f"Target layer '{target_layer_name}' not found.")

    image_tensor.requires_grad_()
    output = model(image_tensor)
    class_idx = output.argmax()
    output[0, class_idx].backward()

    if grad is None or activation is None:
        raise RuntimeError("Grad-CAM failed: Activation or Gradient not captured.")

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1).squeeze()
    cam = torch.clamp(cam, min=0)
    cam = cam / (cam.max() + 1e-8)  # prevent division by zero
    return cam.cpu().numpy()

# ---------- PREDICT ----------
def predict(image, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_index = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_index].item()

    return class_names[pred_index], confidence, img_tensor

# ---------- MAIN STREAMLIT APP ----------
def main():
    st.set_page_config(page_title="🧠 Brain Tumor Classifier", layout="centered")

    # ✅ Display cover image from assets folder
    cover_img_path = "assets/img1.png"
    if os.path.exists(cover_img_path):
        st.image(cover_img_path, use_container_width=True)
    else:
        st.warning("⚠️ Cover image not found in assets folder.")

    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5rem; color: #e91e63; text-align: center; margin-top: 1rem;
        }
        .sub-text {
            text-align: center; font-size: 1.1rem; color: #444; margin-bottom: 2rem;
        }
        .footer {
            position: fixed; bottom: 10px; width: 100%;
            text-align: center; color: gray; font-size: 0.8rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">🧠 Brain Tumor MRI Image Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-text">Upload a brain MRI image to predict the tumor type.</p>', unsafe_allow_html=True)

    model_type = st.radio("📌 Choose Model", ["ResNet50", "Custom CNN"], horizontal=True)
    model, class_names = load_model(model_type)

    uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Uploaded MRI Image", use_container_width=True)

        if st.button("🔍 Classify Tumor"):
            with st.spinner("⏳ Predicting tumor type..."):
                pred_class, confidence, img_tensor = predict(image, model, class_names)

                st.success(f"🎯 Predicted Tumor Type: **{pred_class}**")
                st.info(f"📊 Confidence Score: {confidence * 100:.2f}%")

                # Grad-CAM Heatmap
                try:
                    st.subheader("🔥 Grad-CAM Visualization")
                    target_layer = "layer4" if model_type == "ResNet50" else "conv1"
                    cam = apply_grad_cam(model, img_tensor, target_layer_name=target_layer)

                    # Normalize heatmap and resize
                    heatmap = Image.fromarray(np.uint8(255 * cam)).resize(image.size).convert("L")
                    heatmap_np = np.array(heatmap)

                    # Overlay heatmap on original image
                    image_np = np.array(image).astype(float)
                    heatmap_colored = np.stack([heatmap_np]*3, axis=2).astype(float)

                    overlay = image_np * 0.5 + heatmap_colored * 0.5
                    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

                    st.image(overlay, caption="🧠 Grad-CAM Heatmap Overlay", use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Grad-CAM unavailable: {str(e)}")

                # Download Report
                output_text = f"Predicted Tumor Type: {pred_class}\nConfidence: {confidence*100:.2f}%"
                buffer = io.BytesIO()
                buffer.write(output_text.encode())
                buffer.seek(0)
                st.download_button("📥 Download Prediction Report", buffer, file_name="prediction.txt")

    st.markdown('<div class="footer">Made with ❤️ for Brain Tumor Detection Project</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
