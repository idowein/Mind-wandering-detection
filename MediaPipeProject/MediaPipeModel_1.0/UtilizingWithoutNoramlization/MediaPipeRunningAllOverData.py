import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, messagebox, Label, ttk
import os
import csv

# Initialize MediaPipe Face Mesh and Face Detection
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.7
)

def process_videos_in_directory(directory_path, csv_output_path):
    # List all .mp4 files in the directory and subdirectories
    video_paths = []
    for root_dir, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_paths.append(os.path.join(root_dir, file))

    total_videos = len(video_paths)
    if total_videos == 0:
        messagebox.showwarning("No Videos Found", "No .mp4 files found in the specified directory.")
        root.destroy()
        return

    # Prepare CSV file
    csv_columns = ['Video', 'Blinking', 'Eyebrow Contractions', 'Cheek Raising', 'Facial Twitching', 'Yawning', 'Smiling', 'Frowning', 'Looking Away']
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    csv_file = open(csv_output_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()

    # Update progress bar maximum
    progress_bar['maximum'] = total_videos

    # Process each video
    for idx, video_path in enumerate(video_paths):
        result = process_video(video_path)
        writer.writerow(result)

        # Update progress bar
        progress_bar['value'] = idx + 1
        progress_label.config(text=f"Processing video {idx + 1}/{total_videos}")
        root.update_idletasks()

    csv_file.close()
    messagebox.showinfo("Processing Complete", f"Processing complete. Results saved to {csv_output_path}")
    root.destroy()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    counts = {
        'Video': os.path.basename(video_path),
        'Blinking': 0,
        'Eyebrow Contractions': 0,
        'Cheek Raising': 0,
        'Facial Twitching': 0,
        'Yawning': 0,
        'Smiling': 0,
        'Frowning': 0,
        'Looking Away': 0
    }

    # States to prevent multiple counts for continuous detections
    states = {
        'Blinking': False,
        'Eyebrow Contractions': False,
        'Cheek Raising': False,
        'Facial Twitching': False,
        'Yawning': False,
        'Smiling': False,
        'Frowning': False,
        'Looking Away': False
    }

    prev_landmarks = None
    prev_movement = 0  # For facial twitching

    # Buffers for moving averages
    ear_buffer = []
    mar_buffer = []
    lip_distance_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mesh = face_mesh.process(image_rgb)
        results_detection = face_detection.process(image_rgb)

        if results_mesh.multi_face_landmarks:
            face_landmarks = results_mesh.multi_face_landmarks[0].landmark

            # Blinking detection
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            left_ear = compute_ear(face_landmarks, left_eye_indices)
            right_ear = compute_ear(face_landmarks, right_eye_indices)
            ear = (left_ear + right_ear) / 2.0
            ear_buffer.append(ear)
            if len(ear_buffer) > 5:
                ear_buffer.pop(0)
            ear_avg = np.mean(ear_buffer)
            blink_threshold = 0.25  # Adjusted threshold
            if ear_avg < blink_threshold:
                if not states['Blinking']:
                    counts['Blinking'] += 1
                    states['Blinking'] = True
            else:
                states['Blinking'] = False

            # Eyebrow Contractions detection
            eyebrow_indices = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
            eyebrow_positions = [face_landmarks[i].y for i in eyebrow_indices]
            eyebrow_movement = np.std(eyebrow_positions)
            eyebrow_threshold = 0.01  # Adjusted threshold
            if eyebrow_movement > eyebrow_threshold:
                if not states['Eyebrow Contractions']:
                    counts['Eyebrow Contractions'] += 1
                    states['Eyebrow Contractions'] = True
            else:
                states['Eyebrow Contractions'] = False

            # Cheek Raising detection
            cheek_indices = [50, 101, 118, 280, 352, 429]
            cheek_positions = [face_landmarks[i].y for i in cheek_indices]
            cheek_movement = np.std(cheek_positions)
            cheek_threshold = 0.01  # Adjusted threshold
            if cheek_movement > cheek_threshold:
                if not states['Cheek Raising']:
                    counts['Cheek Raising'] += 1
                    states['Cheek Raising'] = True
            else:
                states['Cheek Raising'] = False

            # Facial Twitching detection
            if prev_landmarks is not None:
                movements = [np.linalg.norm(
                    np.array([face_landmarks[i].x, face_landmarks[i].y]) -
                    np.array([prev_landmarks[i].x, prev_landmarks[i].y])
                ) for i in range(len(face_landmarks))]
                movement = np.mean(movements)
                twitch_threshold = prev_movement + 0.0005  # Adjusted threshold
                if movement > twitch_threshold:
                    if not states['Facial Twitching']:
                        counts['Facial Twitching'] += 1
                        states['Facial Twitching'] = True
                else:
                    states['Facial Twitching'] = False
                prev_movement = movement
            prev_landmarks = face_landmarks

            # Yawning detection
            mouth_indices = [61, 291, 81, 178, 13, 14, 17, 0]
            mar = compute_mar(face_landmarks, mouth_indices)
            mar_buffer.append(mar)
            if len(mar_buffer) > 5:
                mar_buffer.pop(0)
            mar_avg = np.mean(mar_buffer)
            mar_threshold = 0.7  # Adjusted threshold
            if mar_avg > mar_threshold:
                if not states['Yawning']:
                    counts['Yawning'] += 1
                    states['Yawning'] = True
            else:
                states['Yawning'] = False

            # Smiling detection
            left_lip_corner = np.array([face_landmarks[61].x, face_landmarks[61].y])
            right_lip_corner = np.array([face_landmarks[291].x, face_landmarks[291].y])
            lip_distance = np.linalg.norm(left_lip_corner - right_lip_corner)
            lip_distance_buffer.append(lip_distance)
            if len(lip_distance_buffer) > 5:
                lip_distance_buffer.pop(0)
            lip_distance_avg = np.mean(lip_distance_buffer)
            smile_threshold = 0.08  # Adjusted threshold
            if lip_distance_avg > smile_threshold:
                if not states['Smiling']:
                    counts['Smiling'] += 1
                    states['Smiling'] = True
            else:
                states['Smiling'] = False

            # Frowning detection
            left_eyebrow_top = face_landmarks[70]
            left_eye_bottom = face_landmarks[145]
            right_eyebrow_top = face_landmarks[300]
            right_eye_bottom = face_landmarks[374]
            left_frown_distance = abs(left_eyebrow_top.y - left_eye_bottom.y)
            right_frown_distance = abs(right_eyebrow_top.y - right_eye_bottom.y)
            frown_distance = (left_frown_distance + right_frown_distance) / 2
            frown_threshold = 0.015  # Adjusted threshold
            if frown_distance < frown_threshold:
                if not states['Frowning']:
                    counts['Frowning'] += 1
                    states['Frowning'] = True
            else:
                states['Frowning'] = False

            # Looking Away Detection
            if results_detection.detections:
                states['Looking Away'] = False
            else:
                if not states['Looking Away']:
                    counts['Looking Away'] += 1
                    states['Looking Away'] = True

        else:
            prev_landmarks = None
            if not states['Looking Away']:
                counts['Looking Away'] += 1
                states['Looking Away'] = True

    cap.release()

    return counts

def compute_ear(landmarks, indices):
    p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
    p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
    p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])

    # Compute distances
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    if C == 0:
        return 0
    ear = (A + B) / (2.0 * C)
    return ear

def compute_mar(landmarks, indices):
    p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
    p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
    p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])
    p7 = np.array([landmarks[indices[6]].x, landmarks[indices[6]].y])
    p8 = np.array([landmarks[indices[7]].x, landmarks[indices[7]].y])

    # Compute distances
    A = np.linalg.norm(p2 - p8)
    B = np.linalg.norm(p3 - p7)
    C = np.linalg.norm(p4 - p6)
    D = np.linalg.norm(p1 - p5)
    if D == 0:
        return 0
    mar = (A + B + C) / (2.0 * D)
    return mar

def setup_gui():
    global root, progress_bar, progress_label

    root = Tk()
    root.title("Mind Wandering Detection")
    root.geometry("600x200")

    # Handle window close event
    def on_closing():
        root.destroy()
        exit(0)  # Ensure the program exits

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Load and display the splash screen image
    logo_path = r'path\to\your\logo.png'  # Update with your logo path
    if os.path.exists(logo_path):
        from PIL import Image, ImageTk
        splash_image = Image.open(logo_path)
        splash_image = splash_image.resize((600, 150), Image.ANTIALIAS)
        splash_photo = ImageTk.PhotoImage(splash_image)

        splash_label = Label(root, image=splash_photo)
        splash_label.pack()

    # Progress label and bar
    progress_label = Label(root, text="")
    progress_label.pack()

    progress_bar = ttk.Progressbar(root, orient='horizontal', mode='determinate', length=500)
    progress_bar.pack(pady=10)

    # Start processing videos after a short delay to ensure GUI loads
    root.after(1000, start_processing)

    root.mainloop()

def start_processing():
    # Set the directory and CSV output paths
    directory_path = r'C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\DataSet\Train'
    csv_output_path = r'C:\Users\idowe\Mind wandering research\data collection\Datasets\DAiSEE dataset\DAiSEE\DAiSEE\MWDetections\results.csv'

    # Start processing videos
    process_videos_in_directory(directory_path, csv_output_path)

if __name__ == "__main__":
    setup_gui()
