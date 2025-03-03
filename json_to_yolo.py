import os
from pathlib import Path
from your_module import your_conversion_function

def process_images_and_labels(images_folder, labels_folder, output_folder, 
                             image_extension='.jpg', label_extension='.txt'):
    """
    Process images and labels:
    - For images with corresponding labels: convert label and save to output folder
    - For images without labels: create empty txt files in the output folder
    
    Args:
        images_folder (str): Path to the folder containing images
        labels_folder (str): Path to the folder containing labels
        output_folder (str): Path to the output folder (folder A)
        image_extension (str): Extension of image files including the dot
        label_extension (str): Extension of label files including the dot
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process all images
    image_count = 0
    with_label_count = 0
    without_label_count = 0
    
    # Get list of all image files
    image_files = [f for f in os.listdir(images_folder) if f.endswith(image_extension)]
    total_images = len(image_files)
    
    print(f"Found {total_images} images to process...")
    
    for i, image_file in enumerate(image_files, 1):
        image_count += 1
        image_base_name = os.path.splitext(image_file)[0]
        label_file_path = os.path.join(labels_folder, image_base_name + label_extension)
        output_label_path = os.path.join(output_folder, image_base_name + label_extension)
        
        # Progress update every 100 images
        if i % 100 == 0 or i == total_images:
            print(f"Processing image {i}/{total_images} ({i/total_images*100:.1f}%)")
        
        # Check if corresponding label exists
        if os.path.exists(label_file_path):
            with_label_count += 1
            # Read label content
            with open(label_file_path, 'r') as f:
                label_content = f.read()
            
            try:
                # Apply custom conversion
                converted_content = your_conversion_function(label_content)
                
                # Write converted content to output folder
                with open(output_label_path, 'w') as f:
                    f.write(converted_content)
            except Exception as e:
                print(f"Error processing label for image {image_file}: {e}")

        else:
            without_label_count += 1
            # Create empty label file
            with open(output_label_path, 'w') as f:
                pass  # Create empty file

    print(f"\nSummary:")
    print(f"Total images processed: {image_count}")
    print(f"Images with labels: {with_label_count}")
    print(f"Images without labels: {without_label_count}")

if __name__ == "__main__":
    # Replace these with your actual folder paths
    IMAGES_FOLDER = "path/to/images_folder"
    LABELS_FOLDER = "path/to/labels_folder"
    OUTPUT_FOLDER = "path/to/folder_A"
    
    # Replace these with your actual file extensions
    IMAGE_EXTENSION = ".jpg"  # or ".png", etc.
    LABEL_EXTENSION = ".txt"  # or ".json", etc.
    
    process_images_and_labels(
        IMAGES_FOLDER, 
        LABELS_FOLDER, 
        OUTPUT_FOLDER,
        IMAGE_EXTENSION,
        LABEL_EXTENSION
    )
