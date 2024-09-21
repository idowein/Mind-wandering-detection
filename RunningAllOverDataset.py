# This code uses the DAiSEE dataset for detecting and analyzing facial expressions related to mind wandering.
# Citation:
# Gupta, P., Gupta, P., & Balasubramanian, V. N. (2016). DAiSEE: Towards User Engagement Recognition in the Wild.
# arXiv preprint arXiv:1609.01885. Retrieved from https://arxiv.org/abs/1609.01885

# This code uses YOLOv8 for object detection.
# Citation:
# Ultralytics, YOLOv8: State-of-the-art object detection and segmentation.
# Retrieved from https://github.com/ultralytics/ultralytics

import os
import cv2
import dlib
import numpy as np
import pandas as pd  # Import pandas for handling CSV files
from ultralytics import YOLO
import subprocess

# Load models
model = YOLO('yolov8n.pt')  # Ensure this model file is in your working directory or provide the full path
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ensure this file is available

# Paths (update these paths as needed)
dataset_dir = r'C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\DataSet'
labels_dir = r'C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\Labels'
output_dir = r'C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Helper function to detect eye blinking and movements
def detect_eye_movements(landmarks):
    left_eye_ratio = (abs(landmarks[37][1] - landmarks[41][1]) + abs(landmarks[38][1] - landmarks[40][1])) / (
            2.0 * abs(landmarks[36][0] - landmarks[39][0]))
    right_eye_ratio = (abs(landmarks[43][1] - landmarks[47][1]) + abs(landmarks[44][1] - landmarks[46][1])) / (
            2.0 * abs(landmarks[42][0] - landmarks[45][0]))

    eye_movement_threshold = 0.2
    is_blinking = left_eye_ratio < eye_movement_threshold and right_eye_ratio < eye_movement_threshold
    return is_blinking

# Function to analyze facial expressions, including eyebrow contraction, cheek raising, and facial twitches
def analyze_facial_expressions(landmarks_points):
    eyebrow_contraction = abs(landmarks_points[21][1] - landmarks_points[22][1]) < 5  # Simplified for illustration
    cheek_raising = landmarks_points[48][1] > landmarks_points[54][1]  # Checking if one side is higher
    facial_twitches = np.var([p[0] for p in landmarks_points]) > 10  # Simplified variance check for twitches

    return eyebrow_contraction, cheek_raising, facial_twitches

# Function to check head movements
def check_head_movement(prev_position, curr_position):
    threshold = 10  # Customizable
    return np.linalg.norm(np.array(prev_position) - np.array(curr_position)) > threshold

# Function to evaluate focus based on the collected data
def evaluate_focus(blink_count, head_movement_count, eyebrow_contraction_count,
                   cheek_raising_count, facial_twitch_count):
    # Define thresholds and weights (These can be adjusted based on empirical data or further research)
    max_blinks = 20  # Maximum blinks expected during a focused period
    max_head_movements = 15  # Maximum head movements expected during focus
    max_eyebrow_contractions = 10  # Example threshold
    max_cheek_raisings = 10  # Example threshold
    max_facial_twitches = 10  # Example threshold

    # Normalize the parameters
    blink_score = max(0, 1 - (blink_count / max_blinks))
    head_movement_score = max(0, 1 - (head_movement_count / max_head_movements))
    eyebrow_contraction_score = max(0, 1 - (eyebrow_contraction_count / max_eyebrow_contractions))
    cheek_raising_score = max(0, 1 - (cheek_raising_count / max_cheek_raisings))
    facial_twitch_score = max(0, 1 - (facial_twitch_count / max_facial_twitches))

    # Combine the scores with weights to calculate the focus percentage
    focus_score = (blink_score * 0.25) + (head_movement_score * 0.25) + (eyebrow_contraction_score * 0.15) + \
                  (cheek_raising_score * 0.15) + (facial_twitch_score * 0.2)
    focus_percentage = focus_score * 100  # Convert to percentage
    return focus_percentage

# Function to convert AVI to MP4 using FFmpeg
def convert_avi_to_mp4(input_file):
    # Check if input file is an avi
    if input_file.endswith('.avi'):
        output_file = input_file.replace('.avi', '.mp4')
        command = f"ffmpeg -i \"{input_file}\" \"{output_file}\""

        try:
            subprocess.run(command, check=True, shell=True)
            print(f"Conversion successful! {output_file} created.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during conversion: {e}")
        except FileNotFoundError:
            print(f"ffmpeg is not installed or not found in the system PATH.")
        return output_file
    else:
        return input_file  # If not an avi file, return the original file path

# Function to process video and analyze parameters
def process_video(video_path):
    try:
        if video_path.endswith('.avi'):
            # Convert .avi to .mp4
            print(f"Converting {video_path} to .mp4...")
            video_path = convert_avi_to_mp4(video_path)

        if not (video_path.endswith('.mp4') or video_path.endswith('.mov')):
            raise ValueError(f"Invalid video file format: {video_path}. Expected .mp4 or .mov.")

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        blink_count = 0
        head_movement_count = 0
        eyebrow_contraction_count = 0
        cheek_raising_count = 0
        facial_twitch_count = 0
        prev_head_position = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Convert frame to RGB (YOLO expects RGB images)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, classes=[0])  # Detecting faces (class 0)

            # If no faces detected by YOLO, skip to next frame
            if not results:
                continue

            # Assuming only one face per frame for simplicity
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                landmarks_points = [(p.x, p.y) for p in landmarks.parts()]

                # Detect eye blinks
                if detect_eye_movements(landmarks_points):
                    blink_count += 1

                # Track head movements
                curr_head_position = np.mean(landmarks_points, axis=0)
                if prev_head_position is not None and check_head_movement(prev_head_position, curr_head_position):
                    head_movement_count += 1
                prev_head_position = curr_head_position

                # Analyze facial expressions
                eyebrow_contraction, cheek_raising, facial_twitches = analyze_facial_expressions(landmarks_points)
                eyebrow_contraction_count += int(eyebrow_contraction)
                cheek_raising_count += int(cheek_raising)
                facial_twitch_count += int(facial_twitches)

            if frame_count >= 100:  # Analyze first 100 frames for simplicity
                break
        cap.release()

        # Evaluate the student's focus
        focus_percentage = evaluate_focus(blink_count, head_movement_count, eyebrow_contraction_count,
                                          cheek_raising_count, facial_twitch_count)
        return {
            'Video': os.path.basename(video_path),
            'Blink_Count': blink_count,
            'Head_Movements': head_movement_count,
            'Eyebrow_Contractions': eyebrow_contraction_count,
            'Cheek_Raisings': cheek_raising_count,
            'Facial_Twitches': facial_twitch_count,
            'Estimated_Focus': focus_percentage
        }

    except Exception as e:
        print(f"An error occurred while processing {video_path}: {e}")
        # This will skip this video and continue with the next
        return None

# Function to process all videos in the dataset directory
def process_all_videos():
    results_list = []

    # Walk through the dataset directory
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.avi'):
                video_path = os.path.join(root, file)
                print(f"Processing video: {video_path}")
                result = process_video(video_path)
                if result:
                    results_list.append(result)

    # Save results to CSV
    if results_list:
        df = pd.DataFrame(results_list)
        csv_file_path = os.path.join(output_dir, 'facial_expression_results.csv')
        df.to_csv(csv_file_path, index=False)
        print(f"Results have been saved to {csv_file_path}")

        # Now read labels and compute correlation matrix
        compute_correlation(df)
    else:
        print("No results to save.")

# Function to read labels and compute correlation matrix
def compute_correlation(facial_df):
    # Read labels CSV files
    labels_files = [f for f in os.listdir(labels_dir) if f.endswith('.csv')]
    labels_df_list = []

    for labels_file in labels_files:
        labels_path = os.path.join(labels_dir, labels_file)
        temp_df = pd.read_csv(labels_path)
        labels_df_list.append(temp_df)

    # Concatenate all labels into one DataFrame
    labels_df = pd.concat(labels_df_list, ignore_index=True)

    # Assume that 'ClipID' column in labels_df corresponds to 'Video' in facial_df (adjust as necessary)
    # Modify the labels DataFrame to match the video filenames if necessary
    labels_df['Video'] = labels_df['ClipID'].apply(lambda x: f"{x}.mp4")  # Adjust if needed

    # Merge facial_df and labels_df on 'Video'
    merged_df = pd.merge(facial_df, labels_df, on='Video')

    # Select relevant columns for correlation
    facial_features = ['Blink_Count', 'Head_Movements', 'Eyebrow_Contractions',
                       'Cheek_Raisings', 'Facial_Twitches', 'Estimated_Focus']
    daisee_measures = ['Engagement', 'Boredom', 'Confusion', 'Frustration']

    data_to_correlate = merged_df[facial_features + daisee_measures]

    # Compute correlation matrix
    correlation_matrix = data_to_correlate.corr()

    # Display the correlation matrix
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Save the correlation matrix to a CSV file
    corr_csv_path = os.path.join(output_dir, 'correlation_matrix.csv')
    correlation_matrix.to_csv(corr_csv_path)
    print(f"Correlation matrix saved to {corr_csv_path}")

    # Optionally, visualize the correlation matrix
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix between Facial Expressions and DAiSEE Measures')
        plt.show()
    except ImportError:
        print("Seaborn or Matplotlib not installed. Skipping heatmap visualization.")

# Main function
if __name__ == "__main__":
    process_all_videos()
