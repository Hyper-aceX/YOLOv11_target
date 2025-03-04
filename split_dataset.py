import os
import shutil
import random
from pathlib import Path
import glob
import datetime

# Print script execution info
print(f"Script execution started at: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
print(f"User: Hyper-aceX")

# Set random seed for reproducibility
random.seed(42)

# Configuration
image_ext = ".png"
label_ext = ".txt"
source_images_dir = "images"  # Update this to your images directory path
source_labels_dir = "labels"  # Update this to your labels directory path
output_dir = "dataset"  # Base output directory

# Create destination directories
dest_dirs = {
    'train': {'images': os.path.join(output_dir, 'train', 'images'),
              'labels': os.path.join(output_dir, 'train', 'labels')},
    'val': {'images': os.path.join(output_dir, 'val', 'images'),
            'labels': os.path.join(output_dir, 'val', 'labels')},
    'test': {'images': os.path.join(output_dir, 'test', 'images'),
             'labels': os.path.join(output_dir, 'test', 'labels')}
}

# Create all required directories
for split_type in dest_dirs.values():
    for dir_path in split_type.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

# Get list of all images
all_images = glob.glob(os.path.join(source_images_dir, f"*{image_ext}"))
print(f"Found {len(all_images)} images in total")

# Get list of all labels
all_labels = glob.glob(os.path.join(source_labels_dir, f"*{label_ext}"))
print(f"Found {len(all_labels)} labels in total")

# Extract base names without extensions for all labels
label_basenames = set(os.path.splitext(os.path.basename(label))[0] for label in all_labels)
print(f"Unique label base names: {len(label_basenames)}")

# Find images that have corresponding labels
valid_images = []
for img_path in all_images:
    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    if img_basename in label_basenames:
        valid_images.append(img_path)

print(f"Images with corresponding labels: {len(valid_images)}")

# Shuffle the dataset for randomness
random.shuffle(valid_images)

# Calculate split sizes based on the number of valid images
total = len(valid_images)
train_size = int(0.7 * total)
val_size = int(0.2 * total)
# test_size is the remainder

# Split the dataset
train_images = valid_images[:train_size]
val_images = valid_images[train_size:train_size + val_size]
test_images = valid_images[train_size + val_size:]

print(f"Train set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
print(f"Test set: {len(test_images)} images")

# Function to copy files to destination directories
def copy_files(image_list, split_type):
    copied_count = 0
    
    for img_path in image_list:
        # Get base name without extension
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Define source and destination paths
        src_img = img_path
        src_label = os.path.join(source_labels_dir, f"{img_basename}{label_ext}")
        
        # Skip if label doesn't exist (although we've already filtered for valid images)
        if not os.path.exists(src_label):
            print(f"Warning: Label not found for image {img_path}")
            continue
        
        dst_img = os.path.join(dest_dirs[split_type]['images'], os.path.basename(img_path))
        dst_label = os.path.join(dest_dirs[split_type]['labels'], f"{img_basename}{label_ext}")
        
        # Copy files
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_label, dst_label)
        copied_count += 1
    
    return copied_count

# Copy files to respective directories
print("\nCopying train set...")
train_copied = copy_files(train_images, 'train')

print("Copying validation set...")
val_copied = copy_files(val_images, 'val')

print("Copying test set...")
test_copied = copy_files(test_images, 'test')

print("\nDataset splitting completed successfully!")

# Print statistics
print("\nFinal Dataset Statistics:")
for split_type, dirs in dest_dirs.items():
    img_count = len(os.listdir(dirs['images']))
    label_count = len(os.listdir(dirs['labels']))
    print(f"{split_type} set: {img_count} images, {label_count} labels")

print(f"\nTotal images copied: {train_copied + val_copied + test_copied}")
print(f"Total valid image-label pairs: {len(valid_images)}")
