from ultralytics import YOLO
import cv2
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

model = YOLO("yolov8n.pt")
output_frame = None   # Last processed image frame
video_frames = []     # Store all video frames temporarily
video_fps = 30        # Default FPS
video_size = (1280, 720)  # Default size
video_running = False  # Video detection state

# Function to run detection on Image
def detect_image():
    global output_frame
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_path:
        frame = cv2.imread(file_path)
        results = model(frame)
        annotated_frame = results[0].plot()
        output_frame = annotated_frame.copy()  # Save for manual saving

        cv2.imshow("YOLO Image Detection", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Function to run detection on Video
def detect_video():
    global video_frames, video_fps, video_size, video_running
    video_frames = []
    video_running = True

    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if not file_path:
        return

    cap = cv2.VideoCapture(file_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        results = model(frame)
        annotated_frame = results[0].plot()

        # Counters
        people, bicycle, car, motorcycle = 0, 0, 0, 0
        for box in results[0].boxes:
            cls = int(box.cls[0])
            if cls == 0: people += 1
            elif cls == 1: bicycle += 1
            elif cls == 2: car += 1
            elif cls == 3: motorcycle += 1

        cv2.putText(annotated_frame,
                    f"People: {people} | Bicycle: {bicycle} | Bikes: {motorcycle} | Cars: {car}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        video_frames.append(annotated_frame.copy())  # Save frame temporarily
        cv2.imshow("YOLO Video Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    video_running = False
    messagebox.showinfo("Info", "Video detection finished. Press 'Save Last Output' to save.")

# Function to save last output image or video
def save_output():
    global output_frame, video_frames, video_running
    if video_running:
        messagebox.showwarning("Warning", "Video is still running. Wait until it finishes!")
        return

    if video_frames:  # If video was processed
        save_path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                                 filetypes=[("MP4 files", "*.mp4")])
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, video_fps, (1280, 720))
            for frame in video_frames:
                out.write(frame)
            out.release()
            video_frames = []  # Clear frames after saving
            messagebox.showinfo("Saved", f"Video saved at:\n{save_path}")
    elif output_frame is not None:  # If image was processed
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg")])
        if save_path:
            cv2.imwrite(save_path, output_frame)
            messagebox.showinfo("Saved", f"Image saved at:\n{save_path}")
    else:
        messagebox.showwarning("Warning", "No output to save!")

# ---------------- GUI -----------------
root = Tk()
root.title("YOLO Object Detection System")
root.geometry("500x450")
root.configure(bg="#1e1e1e")

Label(root, text="AI Object Detection", font=("Arial", 22, "bold"), fg="white", bg="#1e1e1e").pack(pady=20)

Button(root, text="Upload Image", command=detect_image, font=("Arial", 14),
       bg="#4CAF50", fg="white", width=25).pack(pady=10)

Button(root, text="Upload Video", command=detect_video, font=("Arial", 14),
       bg="#2196F3", fg="white", width=25).pack(pady=10)

Button(root, text="Save Last Output", command=save_output, font=("Arial", 14),
       bg="#FF9800", fg="white", width=25).pack(pady=10)

Label(root, text="Press Q to close detection window", font=("Arial", 10), fg="yellow", bg="#1e1e1e").pack(pady=20)

root.mainloop()
