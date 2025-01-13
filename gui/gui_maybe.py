import sys
import os
import tkinter as tk

# Add the error detection folder to sys.path by going back to the parent directory 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.append(parent_dir)

# Print the current sys.path for debugging 
#print("Current sys.path:") 
#for path in sys.path: 
#    print(path)

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tkinter import filedialog, ttk
from tkinter import Tk, Canvas, Text, Button, Label, Entry, StringVar, END

# Import the error detection scripts
from greenScripts.greenVectorizedSolution import checkGreenFrame
from greenScripts.magentaScreen import checkMagentaFrame
from greenScripts.dropoutScreen import checkBlackFrame
from Highlights.highlights import checkHighlightsFrame
from Frozen.lagff15 import detect_frozen_frame
from HelperScripts.auto_mask import create_mask
from panoto70.panoto70fcn import checkPano


class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")
        self.root.geometry("1000x700")

        # Create and place the canvas for video playback
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

        # Create and place the error log
        self.error_log = tk.Text(root, height=30, width=40)
        self.error_log.grid(row=0, column=3, padx=5, pady=5)

        # Create and place the video progress bar
        self.progress_label = tk.Label(root, text="0:00 / 0:00")
        self.progress_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        self.progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=640, mode='determinate')
        self.progress_bar.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        # Create and place the Choose File button
        self.choose_button = tk.Button(root, text="Choose File", command=self.choose_file)
        self.choose_button.grid(row=2, column=0, padx=5, pady=5)

        # Create and place the Play button
        self.play_button = tk.Button(root, text="Play", command=self.play_video)
        self.play_button.grid(row=2, column=1, padx=5, pady=5)
        self.play_button.config(state=tk.DISABLED)

        # Create and place the threshold controls
        self.threshold_label = tk.Label(root, text="Green Detection Threshold:")
        self.threshold_label.grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        self.threshold_value = tk.StringVar(value="100")  # Default threshold value
        self.threshold_entry = tk.Entry(root, textvariable=self.threshold_value)
        self.threshold_entry.grid(row=2, column=2, padx=5, pady=5)
        self.update_threshold_button = tk.Button(root, text="Update Threshold", command=self.update_threshold)
        self.update_threshold_button.grid(row=2, column=2, padx=5, pady=5, sticky=tk.E)

        # Create and place the status label
        self.status_label = tk.Label(root, text="No video detected.\nRemaining time: N/A", fg="red")
        self.status_label.grid(row=2, column=3, padx=5, pady=5)

        self.cap = None
        self.image_on_canvas = None
        self.play_flag = False
        self.file_path = None
        self.frozen_frame_flags = []
        self.start_time = None
        self.fps = 0

    def update_status(self, message, remaining_time, color="black"):
        self.status_label.config(text=f"{message}\nRemaining time: {remaining_time}", fg=color)

    def update_log(self, message, color="black"):
        self.error_log.insert(tk.END, message + '\n')
        self.error_log.see(tk.END)  # Scroll to the end

    def choose_file(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            if os.path.splitext(self.file_path)[1].lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                self.update_status("Video is currently loaded.", "N/A", "green")
                self.play_button.config(state=tk.NORMAL)
            else:
                self.update_status("Improper file format.", "N/A", "red")
                self.play_button.config(state=tk.DISABLED)
        else:
            self.update_status("No file chosen.", "N/A", "red")

    def play_video(self):
        if self.file_path:
            self.cap = cv2.VideoCapture(self.file_path)
            if not self.cap.isOpened():
                self.update_status("Error with file.", "N/A", "red")
                return
            self.update_status("Video is currently playing.", "Calculating...", "green")
            self.play_flag = True
            self.start_time = time.time()
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frozen_frame_flags = []
            self.process_video_frame()
        else:
            self.update_status("No video detected.", "N/A", "red")

    def process_video_frame(self):
        if self.play_flag and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                self.detect_and_display_errors(frame)

                # Update the progress bar and check if the video is playing in real time
                current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                total_time = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
                self.progress_bar['value'] = (current_time / total_time) * 100
                self.progress_label.config(text=f"{int(current_time//60)}:{int(current_time%60):02d} / {int(total_time//60)}:{int(total_time%60):02d}")

                # Check if video is playing in real time and calculate time discrepancy
                elapsed_real_time = time.time() - self.start_time
                speed_ratio = current_time / elapsed_real_time
                percentage = speed_ratio * 100

                if abs(elapsed_real_time - current_time) < 1:
                    self.update_status(f"Video playing in real time. ({percentage:.2f}%)", "N/A", "green")
                else:
                    remaining_real_time = (total_time - current_time) / speed_ratio
                    self.update_status(f"Video not playing in real time. ({percentage:.2f}%)", f"{int(remaining_real_time//60)}:{int(remaining_real_time%60):02d}", "red")

                self.root.after(25, self.process_video_frame)
            else:
                self.cap.release()
                self.update_status("Finished playing video.", "0:00", "blue")
                self.canvas.delete("all")

    def detect_and_display_errors(self, frame):
        error_display = np.zeros_like(frame)
        error_text = "No Error"

        # Example of integrating error detection scripts
        green_state = checkGreenFrame(frame)
        if green_state:
            error_text = f"Green Screen Error Detected"
            self.update_log(error_text, "red")

        magenta_state = checkMagentaFrame(frame)
        if magenta_state:
            error_text = f"Magenta Screen Error Detected"
            self.update_log(error_text, "red")

        black_state = checkBlackFrame(frame)
        if black_state:
            error_text = f"Dropout Error Detected"
            self.update_log(error_text, "red")

        if checkHighlightsFrame(frame):
            error_text = f"Highlight Shimmer Detected"
            self.update_log(error_text, "red")

        # Assume prev_frame is stored for frozen frame detection
        if hasattr(self, 'prev_frame') and detect_frozen_frame(self.prev_frame, frame):
            error_text = f"Frozen Frame Detected"
            self.update_log(error_text, "red")
            self.frozen_frame_flags.append(1)
        else:
            self.frozen_frame_flags.append(0)

        # Example for checking Pano errors
        lmask, smask = create_mask(frame)
        pano_state = checkPano(frame, smask, lmask)
        if pano_state:
            error_text = f"Pano-70 Error Detected"
            self.update_log(error_text, "red")

        # Update previous frame for frozen frame detection
        self.prev_frame = frame

        # Display error frame in the canvas
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=img_tk)
        self.canvas.image = img_tk  # Store reference to avoid garbage collection

    def update_threshold(self):
        threshold = self.threshold_value.get()
        self.update_status(f"Threshold updated to {threshold}.", "N/A", "blue")
        # Update the threshold value in the error detection script here

def main():
    root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
