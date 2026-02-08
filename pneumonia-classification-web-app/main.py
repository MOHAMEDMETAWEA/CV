import streamlit as st
try:
    import tf_keras as keras
except ImportError:
    import tensorflow.keras as keras

from tf_keras.models import load_model, Model
from tf_keras.layers import DepthwiseConv2D
from PIL import Image
import numpy as np

from util import classify, set_background

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove 'groups' if present, as it might cause issues in some versions
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# Set page config
st.set_page_config(page_title="Pneumonia Classification", page_icon="🫁", layout="wide")

# Add custom CSS for premium medical aesthetics
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e0f2fe 0%, #f8fafc 100%);
    }
    .main {
        padding: 2rem;
    }
    .stHeader {
        background-color: transparent;
    }
    h1 {
        text-align: center;
        color: #0c4a6e;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    h2, h3 {
        color: #075985;
    }
    .stButton>button {
        background-color: #0284c7;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0369a1;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
    }
    .uploaded-img {
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    /* Card-like container for results */
    div[data-testid="stVerticalBlock"] > div:has(h2) {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)


# Load model
model_path = './model/pneumonia_classifier.h5'

@st.cache_resource
def load_classifier(model_path):
    """
    Load the Keras model and cache it to avoid reloading on every interaction.
    """
    try:
        # Using custom_objects for DepthwiseConv2D and loading via tf_keras (Legacy Keras 2)
        # matches the model's original training environment.
        model = load_model(model_path, compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_classifier(model_path)

if model is None:
    st.stop()


# Load class names
try:
    with open('./model/labels.txt', 'r') as f:
        class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
except FileNotFoundError:
    st.error("Labels file not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading labels: {e}")
    st.stop()


# Sidebar
st.sidebar.title("App Info")
st.sidebar.info("""
This app uses a Convolutional Neural Network (CNN) to classify chest X-ray images into two categories:
- **PNEUMONIA**
- **NORMAL**

**Instructions:**
1. Upload a chest X-ray image (JPEG, JPG, PNG).
2. The model will analyze the image.
3. View the prediction and confidence score.
""")
st.sidebar.markdown("---")
st.sidebar.text("Built with Streamlit & Keras")


# Main content
st.title('Pneumonia Classification 🫁')
st.markdown("### Upload a chest X-ray image for analysis")

file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

if file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])

    with col1:
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Uploaded X-ray', use_column_width=True)

    with col2:
        st.markdown("### Analysis Result:")
        with st.spinner('Classifying...'):
            # classify image
            class_name, conf_score = classify(image, model, class_names)

            # Determine color based on prediction
            if class_name == "PNEUMONIA":
                result_color = "red"
                emoji = "⚠️"
            else:
                result_color = "green"
                emoji = "✅"

            st.markdown(f"<h2 style='color: {result_color};'>{emoji} Prediction: {class_name}</h2>", unsafe_allow_html=True)
            
            # Create a progress bar for confidence
            st.progress(float(conf_score))
            st.write(f"### Confidence Score: **{conf_score * 100:.2f}%**")
            
            if class_name == "PNEUMONIA":
                st.warning("The model has detected signs of Pneumonia with high confidence. Please consult a medical professional.")
            else:
                st.success("The model has classified this X-ray as Normal.")
