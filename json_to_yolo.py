import json
import os
from PIL import Image

def convert_json_to_yolo(json_dir, image_dir, output_dir):
    """
    Converts JSON annotation files to YOLOv11 TXT format.

    Args:
        json_dir: Directory containing the JSON annotation files.
        image_dir: Directory containing the images.
        output_dir: Directory to save the YOLO TXT files.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for json_file in os.listdir(json_dir):
        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(json_dir, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract image filename and dimensions
        image_filename = data["image_filename"]  # Adapt this to your JSON structure
        image_path = os.path.join(image_dir, image_filename)
        try:
            image = Image.open(image_path)
            image_width, image_height = image.size
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            continue

        # Create YOLO TXT file
        txt_filename = os.path.splitext(json_file)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, "w") as txt_file:
            for annotation in data["annotations"]:  # Adapt this to your JSON structure
                class_id = annotation["class_id"]  # Adapt this to your JSON structure
                bbox = annotation["bbox"]  # Adapt this to your JSON structure (x, y, width, height)
                keypoints = annotation["keypoints"]  # Adapt this to your JSON structure (list of x, y coordinates)

                # Normalize bounding box coordinates
                x_center = (bbox[0] + bbox[2] / 2) / image_width
                y_center = (bbox[1] + bbox[3] / 2) / image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height

                # Normalize keypoint coordinates
                normalized_keypoints = []
                for x, y in keypoints:
                    normalized_keypoints.append(x / image_width)
                    normalized_keypoints.append(y / image_height)

                # Write to TXT file
                line = f"{class_id} {x_center} {y_center} {width} {height} " + " ".join(map(str, normalized_keypoints))
                txt_file.write(line + "\n")

        print(f"Converted {json_file} to {txt_filename}")

# Example usage:
json_dir = "path/to/your/json/annotations"  # Replace with the path to your JSON annotations
image_dir = "path/to/your/images"  # Replace with the path to your images
output_dir = "path/to/your/yolo/labels"  # Replace with the desired output directory for YOLO TXT files
convert_json_to_yolo(json_dir, image_dir, output_dir)
