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
import pandas as pd
from ultralytics import YOLO
from tkinter import Tk, filedialog, messagebox, Button, Text, END
import subprocess
from PIL import Image, ImageTk
from tkinter import Label

# Load models
model = YOLO('yolov8n.pt')  # YOLO model for face detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Directory to save the output results
output_dir = r'path\to\save\results'  # Update with your desired path

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to convert AVI to MP4 using FFmpeg
def convert_avi_to_mp4(input_file):
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
        return input_file

# Function to process video and count facial expressions
def process_video(video_path):
    if video_path.endswith('.avi'):
        print("Converting .avi to .mp4...")
        video_path = convert_avi_to_mp4(video_path)

    cap = cv2.VideoCapture(video_path)

    # Initialize variables
    measurements = {
        'blink': [],
        'eyebrow_movement': [],
        'cheek_movement': [],
        'gaze_aversion': [],
        'yawn': [],
        'head_nodding': [],
        'lip_biting': [],
        'eye_closure_duration': []
    }

    # Parameters for optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS |
                               cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Variables for optical flow
    prev_gray = None
    prev_landmarks = None
    frame_count = 0

    # Variables for eye closure detection
    eye_closed_frames = 0
    eye_closure_threshold = 3  # Number of consecutive frames to consider as prolonged eye closure
    eye_closures = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        if len(faces) > 0:
            face = faces[0]  # Assuming the first detected face is the subject
            landmarks = predictor(gray, face)
            landmarks_points = np.array([(p.x, p.y) for p in landmarks.parts()], np.float32)

            # Eye Blink Detection
            left_eye_aspect_ratio = compute_eye_aspect_ratio(landmarks_points[36:42])
            right_eye_aspect_ratio = compute_eye_aspect_ratio(landmarks_points[42:48])
            ear = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2.0
            measurements['blink'].append(ear)

            # Eye Closure Detection
            if ear < 0.21:  # Eye closure threshold
                eye_closed_frames += 1
            else:
                if eye_closed_frames >= eye_closure_threshold:
                    eye_closures += 1
                eye_closed_frames = 0

            # Gaze Aversion Detection
            gaze_direction = get_gaze_direction(landmarks_points)
            measurements['gaze_aversion'].append(gaze_direction)

            # Lip Biting Detection
            lip_distance = compute_lip_distance(landmarks_points)
            measurements['lip_biting'].append(lip_distance)

            # Yawning Detection
            mouth_aspect_ratio = compute_mouth_aspect_ratio(landmarks_points[60:68])
            measurements['yawn'].append(mouth_aspect_ratio)

            # Optical Flow for Eyebrow and Cheek Movement
            if prev_gray is not None and prev_landmarks is not None:
                # Calculate optical flow
                curr_points, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_landmarks, None, **lk_params)

                # Eyebrow Movement
                eyebrow_movement = np.linalg.norm(curr_points[17:27] - prev_landmarks[17:27], axis=1).mean()
                measurements['eyebrow_movement'].append(eyebrow_movement)

                # Cheek Movement
                cheek_movement = np.linalg.norm(curr_points[1:16] - prev_landmarks[1:16], axis=1).mean()
                measurements['cheek_movement'].append(cheek_movement)

                # Head Nodding Detection
                head_movement = np.linalg.norm(curr_points - prev_landmarks, axis=1).mean()
                measurements['head_nodding'].append(head_movement)

            prev_gray = gray.copy()
            prev_landmarks = landmarks_points.reshape(-1, 1, 2)

        else:
            # No face detected in frame
            prev_gray = None
            prev_landmarks = None

    cap.release()

    # Optimize thresholds to get counts between 0 and 5
    counts = {}
    thresholds = {}
    desired_count = 5

    # Blink Detection (Using Eye Aspect Ratio)
    blink_threshold = find_optimal_threshold(measurements['blink'], desired_count, lower=True)
    total_blinks = sum(1 for ear in measurements['blink'] if ear < blink_threshold)
    counts['Total Blinks'] = min(total_blinks, 5)
    thresholds['Blink Threshold'] = blink_threshold

    # Eyebrow Movement Detection
    eyebrow_threshold = find_optimal_threshold(measurements['eyebrow_movement'], desired_count, lower=False)
    total_eyebrow_movements = sum(1 for mv in measurements['eyebrow_movement'] if mv > eyebrow_threshold)
    counts['Eyebrow Movements'] = min(total_eyebrow_movements, 5)
    thresholds['Eyebrow Movement Threshold'] = eyebrow_threshold

    # Cheek Movement Detection
    cheek_threshold = find_optimal_threshold(measurements['cheek_movement'], desired_count, lower=False)
    total_cheek_movements = sum(1 for mv in measurements['cheek_movement'] if mv > cheek_threshold)
    counts['Cheek Movements'] = min(total_cheek_movements, 5)
    thresholds['Cheek Movement Threshold'] = cheek_threshold

    # Gaze Aversion Detection
    gaze_threshold = find_optimal_threshold(measurements['gaze_aversion'], desired_count, lower=False)
    total_gaze_aversions = sum(1 for gaze in measurements['gaze_aversion'] if gaze > gaze_threshold)
    counts['Gaze Aversions'] = min(total_gaze_aversions, 5)
    thresholds['Gaze Aversion Threshold'] = gaze_threshold

    # Yawning Detection
    yawn_threshold = find_optimal_threshold(measurements['yawn'], desired_count, lower=False)
    total_yawns = sum(1 for mar in measurements['yawn'] if mar > yawn_threshold)
    counts['Yawns'] = min(total_yawns, 5)
    thresholds['Yawn Threshold'] = yawn_threshold

    # Head Nodding Detection
    nod_threshold = find_optimal_threshold(measurements['head_nodding'], desired_count, lower=False)
    total_head_nods = sum(1 for nod in measurements['head_nodding'] if nod > nod_threshold)
    counts['Head Nods'] = min(total_head_nods, 5)
    thresholds['Head Nod Threshold'] = nod_threshold

    # Lip Biting Detection
    lip_threshold = find_optimal_threshold(measurements['lip_biting'], desired_count, lower=False)
    total_lip_biting = sum(1 for ld in measurements['lip_biting'] if ld < lip_threshold)
    counts['Lip Biting'] = min(total_lip_biting, 5)
    thresholds['Lip Biting Threshold'] = lip_threshold

    # Eye Closure Duration Detection
    counts['Prolonged Eye Closures'] = min(eye_closures, 5)
    thresholds['Eye Closure Frames Threshold'] = eye_closure_threshold

    # Combine results
    results = {'Video': os.path.basename(video_path)}
    results.update(counts)
    results.update(thresholds)

    return results

# Helper functions for measurements
def compute_eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def compute_mouth_aspect_ratio(mouth_points):
    A = np.linalg.norm(mouth_points[2] - mouth_points[7])
    B = np.linalg.norm(mouth_points[4] - mouth_points[7])
    C = np.linalg.norm(mouth_points[0] - mouth_points[6])
    mar = (A + B) / (2.0 * C)
    return mar

def compute_lip_distance(landmarks):
    upper_lip = landmarks[62]
    lower_lip = landmarks[66]
    distance = np.linalg.norm(upper_lip - lower_lip)
    return distance

def get_gaze_direction(landmarks):
    # Simple estimation: distance between nose tip and pupil centers
    # For accurate gaze estimation, eye tracking is required
    # Here we approximate gaze aversion by movement of eyes from center
    nose_tip = landmarks[33]
    left_pupil = landmarks[42]
    right_pupil = landmarks[39]
    left_distance = np.linalg.norm(nose_tip - left_pupil)
    right_distance = np.linalg.norm(nose_tip - right_pupil)
    gaze_direction = (left_distance + right_distance) / 2.0
    return gaze_direction

def find_optimal_threshold(measurements, desired_count=5, lower=True):
    measurements = sorted(measurements, reverse=not lower)
    if len(measurements) < desired_count:
        return measurements[-1] if lower else measurements[0]
    else:
        return measurements[desired_count - 1]

# Function to browse and select files
def browse_files():
    Tk().withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select Video Files",
        filetypes=(("Video files", "*.avi;*.mp4;*.mov"), ("All files", "*.*"))
    )

    if not file_paths:
        messagebox.showwarning("No File Selected", "Please select at least one video file.")
        return

    result_text.delete('1.0', END)  # Clear previous results

    # Create a list to store results for saving to CSV
    all_results = []

    for video_path in file_paths:
        try:
            # Process the video and gather results
            result = process_video(video_path)
            all_results.append(result)

            # Display results in the text box
            result_text.insert(END, f"Video: {result['Video']}\n")
            for key in result:
                if key != 'Video':
                    result_text.insert(END, f"  {key}: {result[key]}\n")
            result_text.insert(END, "\n")

        except (ValueError, FileNotFoundError) as e:
            messagebox.showerror("Input Error", str(e))

    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv_path = os.path.join(output_dir, 'facial_expression_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to {results_csv_path}")

# Main GUI setup with splash screen functionality
def setup_gui():
    root = Tk()
    root.title("Mind Wandering Detection")
    root.geometry("600x350")

    # Handle window close event
    def on_closing():
        root.destroy()
        exit(0)  # Ensure the program exits

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Load and display the splash screen image
    splash_image = Image.open(r'C:\Users\idowe\PycharmProjects\MindWanderingDetection\MindWanderingDetection logo.png')  # Update with your logo path
    splash_image = splash_image.resize((600, 300), Image.Resampling.LANCZOS)
    splash_photo = ImageTk.PhotoImage(splash_image)

    splash_label = Label(root, image=splash_photo)
    splash_label.pack()

    # Hide splash screen and show main GUI after 3 seconds
    def show_main_gui():
        splash_label.pack_forget()  # Remove splash screen
        # Now show the actual GUI
        browse_button = Button(root, text="Browse Files", command=browse_files, width=20, height=2)
        browse_button.pack(pady=20)

        global result_text
        result_text = Text(root, height=15, width=70)
        result_text.pack()

    # Delay for 3 seconds then show main GUI
    root.after(3000, show_main_gui)

    root.mainloop()

# Run the GUI
if __name__ == "__main__":
    setup_gui()
