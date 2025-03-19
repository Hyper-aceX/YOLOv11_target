# Phase 1: One-Time Setup
## Step 1: Camera Calibration
```python
def calibrate_camera(checkerboard_size=(9, 6), square_size=25.0):
    """Calibrate camera to determine intrinsic parameters"""
    # Prepare object points (0,0,0), (1,0,0), ..., (8,5,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale to actual size in mm
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Capture multiple images of checkerboard
    while len(objpoints) < 15:  # Collect at least 15 good images
        frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            # Draw corners for visualization
            cv2.drawChessboardCorners(frame, checkerboard_size, corners2, ret)
            cv2.imshow('Calibration', frame)
            
            print(f"Captured image {len(objpoints)}/15")
            cv2.waitKey(1000)  # Wait for user to reposition board
        else:
            cv2.imshow('Calibration', frame)
            cv2.waitKey(30)
    
    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"Calibration complete. Mean reprojection error: {mean_error/len(objpoints)}")
    
    # Save calibration results
    np.savez('camera_calibration.npz', 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs,
             image_size=gray.shape[::-1])

    return camera_matrix, dist_coeffs
```

## Step 2: YOLO Model Training/Setup for Board Detection
```python
def setup_yolo_model():
    """Load or train YOLO model for detecting board corners"""
    # Option 1: Use pre-trained YOLOv8 and fine-tune
    model = YOLO('yolov8n.pt')
    
    # Fine-tune for corner detection (if using custom training)
    # Prepare dataset with labeled corners
    # model.train(data='board_corners.yaml', epochs=100, imgsz=640)
    
    # Option 2: Use YOLOv8 detect model directly
    # model = YOLO('yolov8n.pt')
    
    return model
```

## Step 3: Define Board Coordinate System
```python
def define_board_model(width_mm=300, height_mm=200):
    """Define 3D model of the board with origin at center"""
    # Define board corners with origin at center
    BOARD_CORNERS_3D = np.array([
        [-width_mm/2, -height_mm/2, 0],  # Top-left
        [width_mm/2, -height_mm/2, 0],   # Top-right
        [width_mm/2, height_mm/2, 0],    # Bottom-right
        [-width_mm/2, height_mm/2, 0]    # Bottom-left
    ], dtype=np.float32)
    
    return BOARD_CORNERS_3D
```

## Step 4: Camera-to-Barrel Calibration
```python
def calibrate_camera_to_barrel():
    """Calibrate relative position and orientation between camera and barrel"""
    # This typically involves a manual or semi-automated procedure:
    # 1. Place calibration target at known locations
    # 2. For each location:
    #    a. Get camera's view and compute 3D position via PnP
    #    b. Move barrel to point at target
    #    c. Record camera position estimate and barrel position
    
    # For a simple approach, if camera is mounted on barrel with known offset:
    camera_to_barrel = np.eye(4)  # Start with identity matrix
    
    # Set translation offset (X, Y, Z) in mm
    camera_to_barrel[0, 3] = 50   # Example: camera is 50mm right of barrel
    camera_to_barrel[1, 3] = -30  # Example: camera is 30mm above barrel
    camera_to_barrel[2, 3] = 20   # Example: camera is 20mm behind barrel
    
    # You might also have a rotation offset (depends on mounting)
    # This is a simplified example assuming camera and barrel point in same direction
    
    return camera_to_barrel
```

## Step 5: Initialize Kalman Filter
```python
def setup_kalman_filter():
    """Setup Kalman filter for target tracking"""
    # State: [x, y, z, vx, vy, vz] - position and velocity
    kalman = cv2.KalmanFilter(6, 3)
    
    # Transition matrix (physics model)
    kalman.transitionMatrix = np.array([
        [1, 0, 0, 1, 0, 0],  # x = x + vx*dt (dt=1)
        [0, 1, 0, 0, 1, 0],  # y = y + vy*dt
        [0, 0, 1, 0, 0, 1],  # z = z + vz*dt
        [0, 0, 0, 1, 0, 0],  # vx = vx
        [0, 0, 0, 0, 1, 0],  # vy = vy
        [0, 0, 0, 0, 0, 1]   # vz = vz
    ], np.float32)
    
    # Measurement matrix (we only measure position, not velocity)
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0]
    ], np.float32)
    
    # Process noise covariance (how much we trust our motion model)
    kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
    
    # Measurement noise covariance (how much we trust our measurements)
    kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
    
    # Initial state covariance (high uncertainty initially)
    kalman.errorCovPost = np.eye(6, dtype=np.float32) * 10
    
    return kalman
```

# Phase 2: Main Operation Loop
## Step 6: Frame Acquisition and YOLO Detection
```python
def detect_board_corners(frame, yolo_model):
    """Detect board corners using YOLO"""
    # Option 1: Direct corner detection
    results = yolo_model(frame)
    
    # Extract corner detections (implementation depends on YOLO output format)
    corners = []
    for detection in results.xyxy[0]:  # Assuming xyxy format
        if detection[5] == 0:  # Class 0 = top-left
            corners.append(detection[:2].cpu().numpy())
        elif detection[5] == 1:  # Class 1 = top-right
            corners.append(detection[:2].cpu().numpy())
        elif detection[5] == 2:  # Class 2 = bottom-right
            corners.append(detection[:2].cpu().numpy())
        elif detection[5] == 3:  # Class 3 = bottom-left
            corners.append(detection[:2].cpu().numpy())
    
    # Ensure corners are in correct order: [top-left, top-right, bottom-right, bottom-left]
    # Sort corners appropriately if needed
    
    # Option 2: Detect full board and extract corners
    # results = yolo_model(frame)
    # board_box = results.xyxy[0][0].cpu().numpy()[:4]  # x1, y1, x2, y2
    # corners = np.array([
    #     [board_box[0], board_box[1]],  # top-left
    #     [board_box[2], board_box[1]],  # top-right
    #     [board_box[2], board_box[3]],  # bottom-right
    #     [board_box[0], board_box[3]]   # bottom-left
    # ])
    
    return np.array(corners, dtype=np.float32) if len(corners) == 4 else None
```

## Step 7: Solve PnP for Board Position
```python
def solve_board_position(corners_2d, board_corners_3d, camera_matrix, dist_coeffs):
    """Solve PnP to get board position and orientation"""
    if corners_2d is None or len(corners_2d) < 4:
        return None, None, None
    
    # Solve PnP
    success, rotation_vec, translation_vec = cv2.solvePnP(
        board_corners_3d,
        corners_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None, None, None
    
    # Convert rotation vector to matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    
    return success, rotation_matrix, translation_vec
```

## Step 8: Calculate Aiming Parameters
```python
def calculate_aiming_parameters(translation_vec, rotation_matrix=None):
    """Calculate distance, azimuth and elevation for aiming"""
    # Extract position (since board origin is at center, this is the aim point)
    position = translation_vec.reshape(3)
    
    # Calculate distance
    distance = np.linalg.norm(position)
    
    # Calculate azimuth (horizontal angle)
    # Note: azimuth is 0 when target is directly ahead, positive to the right
    azimuth = np.degrees(np.arctan2(position[0], position[2]))
    
    # Calculate elevation (vertical angle)
    # Note: elevation is 0 when target is at same height, positive when target is below
    # (negative when above, due to camera coordinate system convention)
    elevation = np.degrees(np.arctan2(-position[1], np.sqrt(position[0]**2 + position[2]**2)))
    
    return {
        'position': position,
        'distance': distance,
        'azimuth': azimuth,
        'elevation': elevation
    }
```

## Step 9: Apply Kalman Filter for Tracking
```python
def apply_kalman_filter(kalman, position, detected):
    """Apply Kalman filter to smooth position and predict motion"""
    # If no detection, just predict
    if not detected:
        prediction = kalman.predict()
        return prediction[:3].reshape(3)
    
    # Predict next state
    prediction = kalman.predict()
    
    # Update with measurement
    measurement = np.array(position, dtype=np.float32).reshape(3, 1)
    corrected = kalman.correct(measurement)
    
    # Extract position from state
    filtered_position = corrected[:3].reshape(3)
    
    return filtered_position
```

## Step 10: Compensate for Ballistics
```python
def compensate_ballistics(distance, elevation, muzzle_velocity=50.0):
    """Apply ballistic compensation for projectile drop"""
    # Simple ballistic model
    gravity = 9.81  # m/s²
    
    # Convert to radians for calculation
    elevation_rad = np.radians(elevation)
    
    # Time of flight
    t = distance / (muzzle_velocity * np.cos(elevation_rad))
    
    # Vertical drop
    drop = 0.5 * gravity * t * t
    
    # Calculate additional elevation angle needed
    additional_angle = np.degrees(np.arctan(drop / distance))
    
    # Adjusted elevation
    compensated_elevation = elevation + additional_angle
    
    return compensated_elevation
```

## Step 11: Interface with Barrel Control System
```python
def control_barrel(azimuth, elevation):
    """Send commands to barrel control system"""
    # Convert angles to servo positions
    # This depends on your specific hardware implementation
    azimuth_servo_value = map_angle_to_servo(azimuth, 
                                            min_angle=-90, max_angle=90,
                                            min_servo=1000, max_servo=2000)
    elevation_servo_value = map_angle_to_servo(elevation,
                                              min_angle=-45, max_angle=45,
                                              min_servo=1000, max_servo=2000)
    
    # Send to hardware controller
    # Example using PySerial for Arduino control
    command = f"AIM {azimuth_servo_value} {elevation_servo_value}\n"
    serial_port.write(command.encode())
    
    # Wait for acknowledgment if needed
    response = serial_port.readline().decode().strip()
    return response == "OK"

def map_angle_to_servo(angle, min_angle, max_angle, min_servo, max_servo):
    """Map angle value to servo control value"""
    # Ensure angle is within bounds
    angle = max(min_angle, min(angle, max_angle))
    
    # Linear mapping
    servo_range = max_servo - min_servo
    angle_range = max_angle - min_angle
    
    return min_servo + (angle - min_angle) * servo_range / angle_range
```

# Phase 3: Complete Integration
## Step 12: Main Control Loop
```python
def main():
    """Main aimbot control loop"""
    # Setup phase
    print("Setting up camera...")
    camera = cv2.VideoCapture(0)
    
    print("Loading calibration data...")
    calib_data = np.load('camera_calibration.npz')
    camera_matrix = calib_data['camera_matrix']
    dist_coeffs = calib_data['dist_coeffs']
    
    print("Loading YOLO model...")
    yolo_model = setup_yolo_model()
    
    print("Setting up board model...")
    board_corners_3d = define_board_model(300, 200)  # 300x200mm board
    
    print("Loading camera-to-barrel transformation...")
    camera_to_barrel = calibrate_camera_to_barrel()  # Or load from file
```
