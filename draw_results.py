import cv2
from PIL import Image
import os
from json_to_yolo import convert_json_to_yolo  # Assuming you have the convert_json_to_yolo script
from split_dataset import split_dataset  # Assuming you have the split_dataset script

def draw_results(image_path, results, output_path):
    """
    Draws bounding boxes and keypoints on an image using OpenCV.

    Args:
        image_path: Path to the image.
        results: List of dictionaries, where each dictionary contains:
            - "bbox": (x1, y1, x2, y2) bounding box coordinates
            - "keypoints": List of (x, y) keypoint coordinates
        output_path: Path to save the output image.
    """
    img = cv2.imread(image_path)

    for result in results:
        bbox = result["bbox"]
        keypoints = result["keypoints"]

        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

        # Draw keypoints
        for x, y in keypoints:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Red circles

    cv2.imwrite(output_path, img)
    print(f"Saved image with detections to {output_path}")

# Example usage (assuming you have the process_results function from the previous step):
image_dir = "path/to/your/test/images"  # replace with your test image directory
json_dir = "path/to/your/json/annotations" # replace with your json annotations directory
output_dir = "path/to/output/images" # replace with your output image directory
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
# Convert all JSON files to YOLO format
convert_json_to_yolo(json_dir, image_dir, "path/to/temp_labels")
# Split dataset to get the test dataset
split_dataset(image_dir, "path/to/temp_labels", "path/to/temp_dataset")

for image_file in os.listdir("path/to/temp_dataset/test"):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join("path/to/temp_dataset/test", image_file)
        # Replace this with your actual YOLOv11 inference code
        # The following is a placeholder - adapt to your YOLOv11 implementation
        # detections = run_yolov11_inference(image_path)
        # For demonstration, let's create some dummy detections:
        detections = [{"confidence": 0.95, "bbox": [0.5, 0.5, 0.2, 0.3], "keypoints": [(0.4, 0.4), (0.6, 0.4), (0.5, 0.6), (0.3, 0.5), (0.7, 0.5)]}]
        results = process_results(image_path, detections)  # Use the process_results function from before

        output_image_path = os.path.join(output_dir, f"detected_{image_file}")
        draw_results(image_path, results, output_image_path)
"""
#remove the temp files
shutil.rmtree("path/to/temp_labels")
shutil.rmtree("path/to/temp_dataset")
"""
