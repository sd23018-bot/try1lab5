import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, ImageEnhance
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Interactive Image Classification",
    layout="centered"
)

st.title("Interactive CPU-Based Image Classification")
st.write("Upload an image and see the top predictions with interactive confidence filtering.")


device = torch.device("cpu")
st.write(f"Running on device: {device}")


resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet18.eval()
resnet18.to(device)

# Preprocessing transformation
preprocess = models.ResNet18_Weights.DEFAULT.transforms()
labels = models.ResNet18_Weights.DEFAULT.meta["categories"]


# Image upload

uploaded_file = st.file_uploader(
    "Upload an image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

# Confidence threshold slider
confidence_threshold = st.sidebar.slider(
    "Confidence threshold (%)",
    0, 100, 0, step=5
) / 100.0  # convert to 0-1

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    # Preprocess and inference
   
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = resnet18(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

   
    # Top-5 predictions
   
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    top5_labels = [labels[i] for i in top5_catid]
    top5_probs = [float(p) for p in top5_prob]

    # Apply confidence threshold
    filtered_labels = []
    filtered_probs = []
    for lbl, prob in zip(top5_labels, top5_probs):
        if prob >= confidence_threshold:
            filtered_labels.append(lbl)
            filtered_probs.append(prob)

    # Display predictions
    st.subheader("Top Predictions:")
    if len(filtered_labels) == 0:
        st.write("No predictions above the confidence threshold.")
    else:
        for lbl, prob in zip(filtered_labels, filtered_probs):
            st.write(f"{lbl}: {prob*100:.2f}%")

        
        # Bar chart
        
        df = pd.DataFrame({
            "Class": filtered_labels,
            "Probability": filtered_probs
        })
        st.bar_chart(df.set_index("Class"))

   
    st.subheader("Approximate Probability Visualization")
    prob_array = probabilities.numpy()
    prob_map = np.zeros((224, 224), dtype=np.float32)
    # For demo: scale top class probability to entire image
    prob_map[:, :] = float(top5_prob[0])
    prob_image = Image.fromarray(np.uint8(prob_map*255))
    prob_image = prob_image.convert("L").resize(image.size)
    prob_overlay = ImageEnhance.Brightness(prob_image).enhance(0.5)
    heatmap = Image.blend(image, prob_overlay.convert("RGB"), alpha=0.3)
    st.image(heatmap, caption="Overlayed heatmap for top prediction", use_column_width=True)
