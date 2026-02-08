# Pneumonia Classification Web App

This project is a web application built with Streamlit that classifies chest X-ray images to detect the presence of **Pneumonia**. It utilizes a pre-trained deep learning model (Keras/TensorFlow) to perform the classification.

## Features

- **User-Friendly Interface**: Simple drag-and-drop interface for uploading chest X-ray images.
- **Real-Time Classification**: Instantly classifies the uploaded image as either **PNEUMONIA** or **NORMAL**.
- **Confidence Score**: Displays the model's confidence score for the prediction.
- **Custom Background**: Features a custom background for a better visual experience.

## Project Structure

```text
pneumonia-classification-web-app/
├── bgs/                # Directory containing background images
├── model/              # Directory containing the trained model and labels
│   ├── pneumonia_classifier.h5  # The pre-trained Keras model
│   └── labels.txt      # Class labels (0 PNEUMONIA, 1 NORMAL)
├── main.py             # Main Streamlit application script
├── util.py             # Utility functions for image processing and classification
├── requirements.txt    # Python dependencies
├── LICENSE             # License file (MIT)
└── README.md           # Project documentation
```

## Installation

**Install dependencies**:
   Ensure you have Python installed (preferably 3.8+). It is recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

   *Note: The project requires `tensorflow`, `keras`, `streamlit`, `Pillow`, and `numpy`.*

## Usage

1. **Run the application**:

   ```bash
   streamlit run main.py
   ```

2. **Open in Browser**:
   The application will automatically open in your default web browser at `http://localhost:8501`.

3. **Classify Images**:
   - Click explicitly on **"Browse files"** or drag and drop a chest X-ray image (JPEG, JPG, PNG).
   - The app will display the image and the prediction result (PNEUMONIA or NORMAL) along with the confidence score.

## Model Details

- **Model Architecture**: The model is a Convolutional Neural Network (CNN) trained on chest X-ray datasets. It expects input images resized to `224x224` pixels.
- **Preprocessing**: Images are normalized to the range `[-1, 1]`.
- **Labels**:
  - `0`: PNEUMONIA
  - `1`: NORMAL
