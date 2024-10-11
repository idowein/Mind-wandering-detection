import cv2
import mediapipe as mp
import numpy as np
import os
import math
from tqdm import tqdm  # Use the standard terminal-based progress bar
import pandas as pd  # For saving the counters to a CSV file

def process_video(input_video_path):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return None  # Return None if video cannot be opened

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Initialize counters for facial expressions
    counters = {
        'Blinking': 0,
        'Facial Twitching': 0,
        'Smiling': 0,
        'Head Tilt': 0
    }

    # Variables for detection states
    blink_state = False
    smile_state = False
    twitch_state = False
    tilt_state = False

    # Previous landmarks for facial twitching detection
    prev_landmarks = None

    # Thresholds
    BLINK_RATIO_THRESHOLD = 0.325
    SMILE_THRESHOLD = 55  # Adjust based on testing
    TWITCH_MOVEMENT_THRESHOLD = 2.1  # Pixels
    HEAD_TILT_THRESHOLD = 15.0  # Degrees for head tilt

    frame_count = 0

    # For progress within the video processing
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_video_path)}")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to find facial landmarks
        face_results = face_mesh.process(image_rgb)

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]

            # Convert normalized landmarks to pixel coordinates
            h, w, _ = image.shape
            landmarks = [(int(point.x * w), int(point.y * h)) for point in face_landmarks.landmark]

            # Detect blinking
            left_eye_ratio = calculate_eye_aspect_ratio([landmarks[i] for i in [33, 160, 158, 133, 153, 144]])
            right_eye_ratio = calculate_eye_aspect_ratio([landmarks[i] for i in [362, 385, 387, 263, 373, 380]])
            avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

            if avg_eye_ratio < BLINK_RATIO_THRESHOLD:
                if not blink_state:
                    counters['Blinking'] += 1
                    blink_state = True
            else:
                blink_state = False

            # Detect smiling
            if is_smiling(landmarks, SMILE_THRESHOLD):
                if not smile_state:
                    counters['Smiling'] += 1
                    smile_state = True
            else:
                smile_state = False

            # Detect facial twitching
            if prev_landmarks is not None:
                if is_facial_twitching(prev_landmarks, landmarks, TWITCH_MOVEMENT_THRESHOLD):
                    if not twitch_state:
                        counters['Facial Twitching'] += 1
                        twitch_state = True
                else:
                    twitch_state = False
            prev_landmarks = landmarks.copy()

            # Detect head tilt using roll angle from head pose estimation
            yaw, pitch, roll = estimate_head_pose(landmarks, w, h)
            if abs(roll) > HEAD_TILT_THRESHOLD:
                if not tilt_state:
                    counters['Head Tilt'] += 1
                    tilt_state = True
            else:
                tilt_state = False

        else:
            prev_landmarks = None

        frame_count += 1
        pbar.update(1)  # Update progress bar

    pbar.close()  # Close the progress bar

    # Release resources
    cap.release()
    face_mesh.close()

    return counters  # Return the counters for this video

def calculate_eye_aspect_ratio(eye_points):
    # Calculate distances between the vertical landmarks
    A = math.dist(eye_points[1], eye_points[5])
    B = math.dist(eye_points[2], eye_points[4])
    # Calculate distance between the horizontal landmarks
    C = math.dist(eye_points[0], eye_points[3])
    # Eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def estimate_head_pose(landmarks, frame_width, frame_height):
    # 3D model points of facial landmarks (reference model)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # 2D image points from landmarks
    image_points = np.array([
        landmarks[1],    # Nose tip
        landmarks[152],  # Chin
        landmarks[33],   # Left eye left corner
        landmarks[263],  # Right eye right corner
        landmarks[61],   # Left Mouth corner
        landmarks[291]   # Right mouth corner
    ], dtype="double")

    # Camera internals
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    # Assume no lens distortion
    dist_coeffs = np.zeros((4,1))

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to Euler angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = np.hstack((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = euler_angles.flatten()

    # Adjust signs of angles
    pitch = -pitch
    yaw = -yaw
    roll = roll

    return yaw, pitch, roll

def is_smiling(landmarks, threshold):
    # Mouth landmarks
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    upper_lip_top = landmarks[13]
    lower_lip_bottom = landmarks[14]

    # Calculate mouth width and height
    mouth_width = math.dist(left_corner, right_corner)
    mouth_height = math.dist(upper_lip_top, lower_lip_bottom)

    if mouth_height == 0:
        return False  # Avoid division by zero

    # Calculate smile ratio
    smile_ratio = mouth_width / mouth_height

    # Compare smile ratio to threshold
    if smile_ratio > threshold:
        return True
    else:
        return False

def is_facial_twitching(prev_landmarks, current_landmarks, threshold):
    # Compare positions of key landmarks between frames
    movement = 0
    num_points = len(prev_landmarks)
    for i in range(num_points):
        dist = math.dist(prev_landmarks[i], current_landmarks[i])
        movement += dist

    avg_movement = movement / num_points

    if avg_movement > threshold:
        return True
    else:
        return False

if __name__ == "__main__":
    import glob

    # Path to the Train directory
    train_dir = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\DataSet\Train"

    # Output CSV file path
    output_csv_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\video_expression_counts.csv"

    # List to hold results
    results = []

    # Find all mp4 files in the Train directory and its subdirectories
    mp4_files = []
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))

    # Process each video file with a progress bar
    for video_file in tqdm(mp4_files, desc="Processing videos", unit="video"):
        counters = process_video(video_file)
        if counters is not None:
            # Add the video file path and counters to the results
            result = {'video_path': video_file}
            result.update(counters)
            results.append(result)
        else:
            print(f"Skipping video due to read error: {video_file}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
