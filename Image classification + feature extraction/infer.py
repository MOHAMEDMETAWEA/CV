import pickle
import argparse
import os
from img2vec_pytorch import Img2Vec
from PIL import Image

def run_inference(image_path, model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print("Initializing Img2Vec...")
    img2vec = Img2Vec()

    print(f"Processing image: {image_path}")
    try:
        img = Image.open(image_path).convert('RGB')
        features = img2vec.get_vec(img)
        
        prediction = model.predict([features])
        print(f"Prediction: {prediction[0]}")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for weather classification.')
    parser.add_argument('--image', type=str, required=True, help='Path to the image to classify.')
    parser.add_argument('--model', type=str, default='./model.p', help='Path to the trained model file.')
    args = parser.parse_args()

    run_inference(args.image, args.model)