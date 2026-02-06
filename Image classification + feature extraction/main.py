import os
import pickle
import argparse
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(data_dir):
    """
    Trains a Random Forest classifier using features extracted by Img2Vec.
    """
    print("Initializing Img2Vec...")
    img2vec = Img2Vec()

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Dataset structure not found in {data_dir}.")
        print("Please run 'prepare_data.py' first.")
        return

    data = {}
    for j, dir_ in enumerate([train_dir, val_dir]):
        split_name = ['training', 'validation'][j]
        print(f"Extracting features for {split_name} data...")
        features = []
        labels = []
        
        for category in os.listdir(dir_):
            cat_path = os.path.join(dir_, category)
            if not os.path.isdir(cat_path):
                continue
                
            print(f"  Processing category: {category}")
            for img_path in os.listdir(cat_path):
                try:
                    full_img_path = os.path.join(cat_path, img_path)
                    img = Image.open(full_img_path).convert('RGB')
                    img_features = img2vec.get_vec(img)
                    features.append(img_features)
                    labels.append(category)
                except Exception as e:
                    print(f"    Failed to process {img_path}: {e}")

        data[f'{split_name}_data'] = features
        data[f'{split_name}_labels'] = labels

    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(data['training_data'], data['training_labels'])

    # Test performance
    print("Evaluating model...")
    y_pred = model.predict(data['validation_data'])
    score = accuracy_score(data['validation_labels'], y_pred)
    print(f"Validation Accuracy: {score * 100:.2f}%")

    # Save the model
    model_path = './model.p'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image classification model using Img2Vec.')
    parser.add_argument('--data_dir', type=str, default='./data/weather_dataset', help='Path to the structured dataset.')
    args = parser.parse_args()

    train_model(args.data_dir)
