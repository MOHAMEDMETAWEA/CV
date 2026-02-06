import os
import shutil
import random

# Configuration
source_dir = 'dataset2'
target_dir = 'data/weather_dataset'
split_ratio = 0.8  # 80% for training, 20% for validation

def prepare_data():
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} not found.")
        return

    # Categories based on the naming convention in dataset2
    categories = ['cloudy', 'rain', 'shine', 'sunrise']
    
    # Create target directories
    for split in ['train', 'val']:
        for category in categories:
            os.makedirs(os.path.join(target_dir, split, category), exist_ok=True)

    # Group files by category
    files_by_category = {cat: [] for cat in categories}
    for filename in os.listdir(source_dir):
        for cat in categories:
            if filename.startswith(cat):
                files_by_category[cat].append(filename)
                break

    # Split and move files
    for cat, files in files_by_category.items():
        random.shuffle(files)
        split_idx = int(len(files) * split_ratio)
        
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        for f in train_files:
            shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, 'train', cat, f))
        
        for f in val_files:
            shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, 'val', cat, f))
            
    print(f"Data preparation complete! Files are now in {target_dir}")

if __name__ == '__main__':
    prepare_data()
