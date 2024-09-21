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
from tkinter import Tk, filedialog, messagebox, Button, Text, END
import subprocess
from PIL import Image, ImageTk
from tkinter import Label

# Load models
model = YOLO('yolov8n.pt')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Directory to save the output results
output_dir = 'path/to/save/results'

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
            print("Converting .avi to .mp4...")
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
            results = model(frame, classes=[0])  # Detecting faces

            for result in results:
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
                    eyebrow_contraction_count += eyebrow_contraction
                    cheek_raising_count += cheek_raising
                    facial_twitch_count += facial_twitches

            if frame_count >= 100:  # Analyze every 100 frames (for simplicity)
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
        print(f"An error occurred: {e}")
        # This will terminate the program execution immediately if an error is encountered.
        exit(1)

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

    results_list = []  # List to store results for CSV

    for video_path in file_paths:
        try:
            # Convert .avi to .mp4 if necessary
            if video_path.endswith('.avi'):
                video_path = convert_avi_to_mp4(video_path)

            # Process the video and gather results
            result = process_video(video_path)
            results_list.append(result)  # Append result to the list

            # Display results in the text box
            result_text.insert(END, f"Video: {result['Video']}\n")
            result_text.insert(END, f"  Blink Count: {result['Blink_Count']}\n")
            result_text.insert(END, f"  Head Movements: {result['Head_Movements']}\n")
            result_text.insert(END, f"  Eyebrow Contractions: {result['Eyebrow_Contractions']}\n")
            result_text.insert(END, f"  Cheek Raisings: {result['Cheek_Raisings']}\n")
            result_text.insert(END, f"  Facial Twitches: {result['Facial_Twitches']}\n")
            result_text.insert(END, f"  Estimated Focus: {result['Estimated_Focus']:.2f}%\n\n")

        except (ValueError, FileNotFoundError) as e:
            messagebox.showerror("Input Error", str(e))

    # Save results to CSV
    if results_list:
        df = pd.DataFrame(results_list)
        csv_file_path = os.path.join(output_dir, 'facial_expression_results.csv')
        df.to_csv(csv_file_path, index=False)
        messagebox.showinfo("Results Saved", f"Results have been saved to {csv_file_path}")

# Main GUI setup with splash screen functionality
def setup_gui():
    root = Tk()
    root.title("Mind Wandering Detection")
    root.geometry("600x300")

    # Handle window close event
    def on_closing():
        root.destroy()
        exit(0)  # Ensure the program exits

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Load and display the splash screen image
    splash_image = Image.open(r'C:\Users\idowe\PycharmProjects\MindWanderingDetection\MindWanderingDetection logo.png')
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
        result_text = Text(root, height=10, width=60)
        result_text.pack()

    # Delay for 3 seconds then show main GUI
    root.after(3000, show_main_gui)

    root.mainloop()

# Run the GUI
if __name__ == "__main__":
    setup_gui()
