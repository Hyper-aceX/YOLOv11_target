import os
import random
import shutil

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits a dataset into training, validation, and test sets.

    Args:
        image_dir: Directory containing the images.
        label_dir: Directory containing the YOLO TXT label files.
        output_dir: Directory to save the split datasets.
        train_ratio: Ratio of images for the training set.
        val_ratio: Ratio of images for the validation set.
        test_ratio: Ratio of images for the test set.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]  # Add other image extensions if needed
    random.shuffle(images)

    num_images = len(images)
    num_train = int(train_ratio * num_images)
    num_val = int(val_ratio * num_images)

    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]

    def move_files(image_list, dest_dir):
        for image_file in image_list:
            image_path = os.path.join(image_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)

            # Move image
            shutil.copy(image_path, os.path.join(dest_dir, image_file))

            # Move label (if it exists)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(dest_dir, label_file))
            else:
                print(f"Warning: No label file found for {image_file}")

    move_files(train_images, train_dir)
    move_files(val_images, val_dir)
    move_files(test_images, test_dir)

    print("Dataset split complete.")

# Example usage:
image_dir = "path/to/your/images"  # Replace with the path to your images
label_dir = "path/to/your/yolo/labels"  # Replace with the path to your YOLO TXT labels
output_dir = "path/to/your/split/dataset"  # Replace with the desired output directory
split_dataset(image_dir, label_dir, output_dir)
