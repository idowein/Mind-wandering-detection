import cv2
import mediapipe as mp
import numpy as np
import os
import math
from collections import deque

def process_video(input_video_path):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Initialize drawing utilities
    mp_drawing = mp.solutions.drawing_utils

    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Prepare output video path
    dir_name = os.path.dirname(input_video_path)
    base_name = os.path.basename(input_video_path)
    name, ext = os.path.splitext(base_name)
    output_video_path = os.path.join(dir_name, f"{name}_processed.mp4")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize counters for facial expressions
    counters = {
        'Blinking': 0,
        'Smiling': 0,
        'Head Movement': 0
    }

    # Variables for detection states
    blink_state = False
    blink_frame_counter = 0
    smile_state = False
    head_movement_state = False

    # Thresholds and parameters
    BLINK_RATIO_THRESHOLD = None        # Will be updated after baseline is established
    SMILE_THRESHOLD = None              # Will be updated after baseline is established
    CHEEK_THRESHOLD = None              # Will be updated after baseline is established
    EAR_SQUINT_THRESHOLD = None         # Will be updated after baseline is established
    HEAD_MOVEMENT_THRESHOLD = 2.5        # Threshold in pixels for average movement
    MIN_FRAMES_FOR_BLINK = 1            # Minimum consecutive frames for blink
    ALPHA = 0.6                         # Smoothing factor for landmarks
    EAR_HISTORY_SIZE = 5                # Size of the EAR history deque
    SMILE_HISTORY_SIZE = 5              # Size of the smile score history deque
    CHEEK_HISTORY_SIZE = 5              # Size of the cheek distance history deque
    EAR_SQUINT_HISTORY_SIZE = 5         # Size of the EAR squint history deque

    # Initialize variables for smoothing and baseline
    ear_history = deque(maxlen=EAR_HISTORY_SIZE)
    smile_history = deque(maxlen=SMILE_HISTORY_SIZE)
    cheek_history = deque(maxlen=CHEEK_HISTORY_SIZE)
    ear_squint_history = deque(maxlen=EAR_SQUINT_HISTORY_SIZE)
    smoothed_landmarks = None
    baseline_ear_values = []
    baseline_smile_ratios = []
    baseline_cheek_distances = []
    baseline_ear_squint = []
    BASELINE_FRAMES = 30                # Number of frames to establish baseline

    frame_count = 0

    # Eye and mouth indices
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]
    mouth_indices = [61, 291, 13, 14]   # Left corner, right corner, upper lip, lower lip

    # Cheek and eye landmarks for cheek raising detection
    left_cheek_index = 205  # Approximate cheek point under the left eye
    right_cheek_index = 425 # Approximate cheek point under the right eye
    left_eye_lower_index = 145  # Lower eyelid of left eye
    right_eye_lower_index = 374 # Lower eyelid of right eye

    # Key landmarks for head movement detection
    key_landmark_indices = [1, 33, 61, 152, 263, 291]  # Nose tip, left eye corner, left mouth corner, chin, right eye corner, right mouth corner
    prev_key_landmarks = None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Finished processing the video.")
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to find facial landmarks
        face_results = face_mesh.process(image_rgb)

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]

            h, w, _ = image.shape
            landmarks = [(point.x * w, point.y * h) for point in face_landmarks.landmark]

            # Smooth landmarks using exponential moving average
            if smoothed_landmarks is None:
                smoothed_landmarks = landmarks.copy()
            else:
                for i in range(len(landmarks)):
                    x = ALPHA * landmarks[i][0] + (1 - ALPHA) * smoothed_landmarks[i][0]
                    y = ALPHA * landmarks[i][1] + (1 - ALPHA) * smoothed_landmarks[i][1]
                    smoothed_landmarks[i] = (x, y)

            # Draw face landmarks
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))

            # Extract eye landmarks
            left_eye_points = [smoothed_landmarks[i] for i in left_eye_indices]
            right_eye_points = [smoothed_landmarks[i] for i in right_eye_indices]

            # Calculate eye aspect ratio (EAR)
            left_eye_ratio = calculate_eye_aspect_ratio(left_eye_points)
            right_eye_ratio = calculate_eye_aspect_ratio(right_eye_points)

            if left_eye_ratio is not None and right_eye_ratio is not None:
                avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2

                # Collect baseline EAR
                if frame_count < BASELINE_FRAMES:
                    baseline_ear_values.append(avg_eye_ratio)
                else:
                    # Apply temporal filtering
                    ear_history.append(avg_eye_ratio)
                    filtered_ear = sum(ear_history) / len(ear_history)

            else:
                avg_eye_ratio = None  # Handle case where EAR cannot be calculated

            # Calculate smile ratio
            smile_ratio = calculate_smile_ratio(smoothed_landmarks)
            if smile_ratio is not None:
                # Collect baseline smile ratios
                if frame_count < BASELINE_FRAMES:
                    baseline_smile_ratios.append(smile_ratio)
                else:
                    # Apply temporal filtering to smile ratio
                    smile_history.append(smile_ratio)
                    filtered_smile_ratio = sum(smile_history) / len(smile_history)

            # Calculate cheek to eye distances (cheek raising)
            left_cheek_distance = calculate_distance(smoothed_landmarks[left_cheek_index], smoothed_landmarks[left_eye_lower_index])
            right_cheek_distance = calculate_distance(smoothed_landmarks[right_cheek_index], smoothed_landmarks[right_eye_lower_index])
            avg_cheek_distance = (left_cheek_distance + right_cheek_distance) / 2

            if frame_count < BASELINE_FRAMES:
                baseline_cheek_distances.append(avg_cheek_distance)
            else:
                # Apply temporal filtering to cheek distance
                cheek_history.append(avg_cheek_distance)
                filtered_cheek_distance = sum(cheek_history) / len(cheek_history)

            # Calculate EAR squint (for eye squinting during smile)
            if left_eye_ratio is not None and right_eye_ratio is not None:
                avg_ear = (left_eye_ratio + right_eye_ratio) / 2
                if frame_count < BASELINE_FRAMES:
                    baseline_ear_squint.append(avg_ear)
                else:
                    # Apply temporal filtering to EAR
                    ear_squint_history.append(avg_ear)
                    filtered_ear_squint = sum(ear_squint_history) / len(ear_squint_history)

            # After baseline frames, set thresholds
            if frame_count == BASELINE_FRAMES:
                # Set thresholds based on baseline averages
                if len(baseline_ear_values) > 0:
                    BLINK_RATIO_THRESHOLD = sum(baseline_ear_values) / len(baseline_ear_values) * 0.8  # Adjust as needed
                else:
                    print("Warning: No EAR data collected during baseline frames.")
                    BLINK_RATIO_THRESHOLD = 0.3  # Set a default value or handle accordingly

                if len(baseline_smile_ratios) > 0:
                    SMILE_THRESHOLD = sum(baseline_smile_ratios) / len(baseline_smile_ratios) * 1.1    # Adjust as needed
                else:
                    print("Warning: No smile data collected during baseline frames.")
                    SMILE_THRESHOLD = 1.5  # Set a default value or handle accordingly

                if len(baseline_cheek_distances) > 0:
                    CHEEK_THRESHOLD = sum(baseline_cheek_distances) / len(baseline_cheek_distances) * 0.9  # Cheeks move closer to eyes when smiling
                else:
                    print("Warning: No cheek distance data collected during baseline frames.")
                    CHEEK_THRESHOLD = avg_cheek_distance * 0.9  # Set a default value or handle accordingly

                if len(baseline_ear_squint) > 0:
                    EAR_SQUINT_THRESHOLD = sum(baseline_ear_squint) / len(baseline_ear_squint) * 0.9  # EAR decreases when squinting
                else:
                    print("Warning: No EAR squint data collected during baseline frames.")
                    EAR_SQUINT_THRESHOLD = 0.2  # Set a default value or handle accordingly

            if frame_count >= BASELINE_FRAMES:
                # Blink detection logic
                if BLINK_RATIO_THRESHOLD is not None and filtered_ear is not None:
                    if filtered_ear < BLINK_RATIO_THRESHOLD:
                        blink_frame_counter += 1
                    else:
                        if blink_frame_counter >= MIN_FRAMES_FOR_BLINK:
                            counters['Blinking'] += 1
                        blink_frame_counter = 0

                # Smiling detection logic using combined features
                if (SMILE_THRESHOLD is not None and CHEEK_THRESHOLD is not None and EAR_SQUINT_THRESHOLD is not None):
                    smile_detected = False

                    # Check if mouth is in smile position
                    mouth_smile = filtered_smile_ratio > SMILE_THRESHOLD if 'filtered_smile_ratio' in locals() else False

                    # Check if cheeks are raised
                    cheeks_raised = filtered_cheek_distance < CHEEK_THRESHOLD if 'filtered_cheek_distance' in locals() else False

                    # Check if eyes are squinted
                    eyes_squinted = filtered_ear_squint < EAR_SQUINT_THRESHOLD if 'filtered_ear_squint' in locals() else False

                    # Combine features
                    if mouth_smile and (cheeks_raised or eyes_squinted):
                        smile_detected = True

                    if smile_detected:
                        smile_state = True
                    else:
                        if smile_state:
                            counters['Smiling'] += 1
                        smile_state = False

            # Head movement detection
            key_landmarks = [smoothed_landmarks[i] for i in key_landmark_indices]

            if prev_key_landmarks is not None:
                # Compute average movement of key landmarks
                total_movement = 0
                for i in range(len(key_landmarks)):
                    movement = math.hypot(key_landmarks[i][0] - prev_key_landmarks[i][0],
                                          key_landmarks[i][1] - prev_key_landmarks[i][1])
                    total_movement += movement
                avg_movement = total_movement / len(key_landmarks)

                # Head movement detection logic
                if avg_movement > HEAD_MOVEMENT_THRESHOLD:
                    if not head_movement_state:
                        counters['Head Movement'] += 1
                        head_movement_state = True
                else:
                    head_movement_state = False
            else:
                head_movement_state = False

            prev_key_landmarks = key_landmarks.copy()

        else:
            # No face detected in the frame
            prev_key_landmarks = None
            head_movement_state = False

        # Draw the counters on the image
        overlay_counters(image, counters)

        # Write the frame to the output video
        out.write(image)

        frame_count += 1
        # Optionally print progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    # Release resources
    cap.release()
    out.release()
    face_mesh.close()

    print(f"Output video saved at: {output_video_path}")

def calculate_eye_aspect_ratio(eye_points):
    """
    Calculates the Eye Aspect Ratio (EAR) for blink detection.
    EAR is the ratio of the distances between the vertical eye landmarks
    and the distances between the horizontal eye landmarks.

    Args:
        eye_points (list): List of tuples containing the eye landmarks.

    Returns:
        float: EAR value.
    """
    # Calculate distances between the vertical landmarks
    A = math.hypot(eye_points[1][0] - eye_points[5][0], eye_points[1][1] - eye_points[5][1])
    B = math.hypot(eye_points[2][0] - eye_points[4][0], eye_points[2][1] - eye_points[4][1])
    # Calculate distance between the horizontal landmarks
    C = math.hypot(eye_points[0][0] - eye_points[3][0], eye_points[0][1] - eye_points[3][1])
    # Avoid division by zero
    if C == 0:
        return None
    # Eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_smile_ratio(landmarks):
    """
    Calculates a smile ratio based on mouth width and height.
    A genuine smile involves an increase in mouth width relative to mouth height.

    Args:
        landmarks (list): List of tuples containing the facial landmarks.

    Returns:
        float: Smile ratio.
    """
    # Mouth landmarks
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    upper_lip_top = landmarks[13]
    lower_lip_bottom = landmarks[14]

    # Calculate mouth width and height
    mouth_width = math.hypot(left_corner[0] - right_corner[0], left_corner[1] - right_corner[1])
    mouth_height = math.hypot(upper_lip_top[0] - lower_lip_bottom[0], upper_lip_top[1] - lower_lip_bottom[1])

    if mouth_height == 0:
        return None  # Avoid division by zero

    # Calculate smile ratio
    smile_ratio = mouth_width / mouth_height
    return smile_ratio

def calculate_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        point1 (tuple): (x, y) coordinates of the first point.
        point2 (tuple): (x, y) coordinates of the second point.

    Returns:
        float: Distance between the two points.
    """
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def overlay_counters(image, counters):
    """
    Draws a semi-transparent rectangle and overlays the expression counters
    on the video frame.

    Args:
        image (numpy.ndarray): The video frame.
        counters (dict): Dictionary containing expression counts.
    """
    # Create a semi-transparent rectangle
    overlay = image.copy()
    h, w, _ = image.shape
    # Adjust the rectangle dimensions
    cv2.rectangle(overlay, (w - 220, 10), (w - 10, 140), (0, 0, 0), -1)
    alpha = 0.5
    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Write the counters
    y0, dy = 40, 25
    for i, (key, value) in enumerate(counters.items()):
        y = y0 + i * dy
        text = f"{key}: {value}"
        cv2.putText(image_new, text, (w - 210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Copy the modified image back
    image[:, :, :] = image_new

if __name__ == "__main__":
    # Replace 'path_to_your_video.mp4' with the path to your input video file
    input_video_path = r"C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\DataSet\Train\110002\1100021001\1100021001.mp4"
    process_video(input_video_path)
