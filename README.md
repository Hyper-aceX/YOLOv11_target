# YOLOv11 Training Guide for Custom Dataset with Keypoints

This guide outlines the steps to train a YOLOv11 object detection model with a custom dataset, including converting JSON labels, splitting the dataset, training the model, and visualizing the results with OpenCV.

## Step 1: Prepare the Dataset

### 1.1: Convert JSON Labels to YOLOv11 TXT Format

YOLOv11 expects labels in a specific TXT format. This script converts JSON labels to the required format. Each TXT file corresponds to an image and contains one line per object, with class ID, normalized bounding box coordinates, and normalized keypoint coordinates.

**Important Adaptations:**

*   **`data["image_filename"]`:**  Modify this to correctly extract the image filename from your JSON.
*   **`data["annotations"]`:**  Modify this to correctly access the list of annotations in your JSON.
*   **`annotation["class_id"]`:** Modify this to correctly extract the class ID from each annotation.
*   **`annotation["bbox"]`:**  Modify this to correctly extract the bounding box coordinates (x, y, width, height) from each annotation.  Make sure the order of coordinates matches the script's expectation.
*   **`annotation["keypoints"]`:**  Modify this to correctly extract the list of keypoint coordinates (x, y pairs) from each annotation.

### 1.2: Split the Dataset into Training, Validation, and Test Sets (7:2:1 Ratio)

This script splits the dataset into training, validation, and test sets. It assumes you have images and corresponding YOLO TXT label files in separate directories.

## Step 2: Train the Model and Process Results

### 2.1: Train the YOLOv11 Model Using the Prepared Dataset

This step involves using the YOLOv11 framework to train your model.

1.  **Install YOLOv11:** Follow the installation instructions in the YOLOv11 documentation. This usually involves installing dependencies like PyTorch, CUDA, etc.
2.  **Configure the YOLOv11 configuration file:** You'll need to modify the YOLOv11 configuration file (`.yaml` or `.cfg` depending on the version) to reflect your dataset:
    *   **Number of classes:** Set this to the number of object classes in your dataset.
    *   **Keypoints:** Configure the model to predict the 5 keypoints you have. This may involve modifying the model architecture.
    *   **Paths to training and validation data:** Specify the paths to the `train` and `val` directories you created in the previous step.
3.  **Start Training:** Use the YOLOv11 training command-line tool or script, providing the path to your configuration file. This will start the training process. Monitor the training progress and adjust hyperparameters as needed.
