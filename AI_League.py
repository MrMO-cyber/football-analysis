import cv2
import numpy as np
from ultralytics import YOLO
from google import generativeai as genai
from collections import Counter
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
import os

# --- الإعدادات ---
GOOGLE_API_KEY = "AIzaSyBw_NPX9ZBBcSG7_-oQXUrhOn0jJw8CkGo"  # استبدل بـ API Key الخاص بك
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
    raise ValueError("API Key not set. Please replace 'YOUR_API_KEY_HERE' with your actual API key.")

genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL_NAME = 'gemini-1.5-flash'

# --- تحميل نموذج YOLOv8 ---
try:
    model_yolo = YOLO("yolov8l.pt")
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

# --- المتغيرات العامة ---
video_path = None
events_log = []

# --- دالة لحساب مركز المربع ---
def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# المتغيرات العامة
video_path = None
events_log = []

# دالة لتحديد ما إذا كان الكائن داخل منطقة الجزاء
def is_goalkeeper(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    center_x, center_y = get_center((x1, y1, x2, y2))
    
    # افترض أن منطقة الجزاء هي 20% من العرض و30% من الطول السفلي
    penalty_box_x_min = int(0.1 * frame_width)
    penalty_box_x_max = int(0.9 * frame_width)
    penalty_box_y_min = int(0.7 * frame_height)
    penalty_box_y_max = frame_height

    # إذا كان اللاعب داخل منطقة الجزاء
    if penalty_box_x_min <= center_x <= penalty_box_x_max and penalty_box_y_min <= center_y <= penalty_box_y_max:
        return True
    return False

# دالة للحصول على اللون السائد للكائن
def get_dominant_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    cropped_frame = frame[y1:y2, x1:x2]
    cropped_frame = cv2.resize(cropped_frame, (50, 50), interpolation=cv2.INTER_AREA)  # تقليل الحجم لتحليل أسرع
    pixels = cropped_frame.reshape(-1, 3)
    most_common_color = Counter([tuple(pixel) for pixel in pixels]).most_common(1)
    return most_common_color[0][0] if most_common_color else None

# --- الوظائف الأصلية ---
# دالة لاختيار ملف الفيديو
def browse_file():
    global video_path, events_log
    events_log = []
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )
    if video_path:
        messagebox.showinfo("File Selected", f"Selected video: {video_path}")

# Helper function to display a frame in the Tkinter label
def show_frame_in_gui(frame):
    try:
        max_height = 480
        h, w = frame.shape[:2]
        if h > max_height:
            ratio = max_height / h
            new_w = int(w * ratio)
            frame = cv2.resize(frame, (new_w, max_height), interpolation=cv2.INTER_AREA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        root.update_idletasks()
        root.update()
    except Exception as e:
        print(f"Error updating GUI frame: {e}")

# دالة لتحليل الفيديو
def analyze_video():
    global video_path, events_log

    if not video_path:
        messagebox.showerror("Error", "No video file selected!")
        return

    events_log = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Unable to open video file: {video_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_filename = "output_analysis.mp4"
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        frame_count = 0
        skip_frames = 1  # تحليل كل إطار لتحسين الدقة

        progress_bar["maximum"] = total_frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            progress_bar["value"] = frame_count
            root.update_idletasks()

            if frame_count % skip_frames != 0:
                continue

            results = model_yolo(frame, verbose=False)

            frame_events = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    label = model_yolo.names[cls]

                    confidence_threshold = 0.5
                    if conf < confidence_threshold:
                        continue

                    if label == "sports ball":
                        ball_center = get_center((x1, y1, x2, y2))
                        frame_events.append(f"Ball detected at {ball_center} (conf: {conf:.2f})")
                        color = (0, 255, 0)
                    elif label == "person":
                        player_center = get_center((x1, y1, x2, y2))
                        dominant_color = get_dominant_color(frame, (x1, y1, x2, y2))

                        if is_goalkeeper((x1, y1, x2, y2), frame_width, frame_height):
                            frame_events.append(f"Goalkeeper detected at {player_center} (color: {dominant_color})")
                            color = (255, 255, 0)  # لون مميز للحارس
                        else:
                            frame_events.append(f"Player detected at {player_center} (conf: {conf:.2f})")
                            color = (255, 0, 0)
                    else:
                        color = (0, 255, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if frame_events:
                events_log.append({"frame": frame_count, "events": frame_events})

            out.write(frame)
            show_frame_in_gui(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        progress_bar["value"] = 0  # Reset progress bar

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during video analysis: {e}")

# --- عرض الملخص ---
def show_summary(summary_text):
    summary_window = tk.Toplevel(root)
    summary_window.title("AI Generated Summary")
    text_widget = tk.Text(summary_window, wrap=tk.WORD, padx=10, pady=10)
    text_widget.pack(expand=True, fill=tk.BOTH)
    text_widget.insert(tk.END, summary_text)
    text_widget.configure(state='disabled')
    close_button = tk.Button(summary_window, text="Close", command=summary_window.destroy)
    close_button.pack(pady=10)

# --- طرح الأسئلة على الذكاء الاصطناعي ---
def ask_ai_question():
    global events_log
    if not events_log:
        messagebox.showinfo("No Data", "No events logged to analyze. Please analyze a video first.")
        return

    question = simpledialog.askstring("Ask AI", "What do you want to know about the match?")
    if not question:
        return

    events_text = "\n".join(
        [f"Frame {event['frame']}: {', '.join(event['events'])}" for event in events_log[:100]]
    )

    prompt = (
        "You are an AI assistant specializing in football video analysis. "
        "Based on the following detected events, answer the user's question accurately and concisely.\n\n"
        f"{events_text}\n\nQuestion: {question}\nAnswer:"
    )

    try:
        model_genai = genai.GenerativeModel(GEMINI_MODEL_NAME)
        generation_config = genai.types.GenerationConfig(max_output_tokens=300, temperature=0.4)
        response = model_genai.generate_content(prompt, generation_config=generation_config)
        answer = response.text.strip() if response.parts else "No answer generated."
        show_answer(answer)

    except Exception as e:
        messagebox.showerror("Error", f"Error generating answer: {e}")

# --- عرض إجابة AI ---
def show_answer(answer_text):
    answer_window = tk.Toplevel(root)
    answer_window.title("AI Answer")
    text_widget = tk.Text(answer_window, wrap=tk.WORD, padx=10, pady=10)
    text_widget.pack(expand=True, fill=tk.BOTH)
    text_widget.insert(tk.END, answer_text)
    text_widget.configure(state='disabled')
    close_button = tk.Button(answer_window, text="Close", command=answer_window.destroy)
    close_button.pack(pady=10)

# --- الواجهة الرسومية ---
root = tk.Tk()
root.title("YOLOv8 Video Analysis with AI Interaction")
root.geometry("800x600")

video_frame = tk.Frame(root)
video_frame.pack(pady=10, padx=10, fill="both", expand=True)
video_label = tk.Label(video_frame, text="Video will appear here after analysis starts")
video_label.pack(fill="both", expand=True)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)
btn_browse = tk.Button(button_frame, text="Browse Video", command=browse_file, width=15, height=2)
btn_browse.pack(side=tk.LEFT, padx=10)
btn_analyze = tk.Button(button_frame, text="Start Analysis & Summary", command=analyze_video, width=25, height=2)
btn_analyze.pack(side=tk.LEFT, padx=10)
btn_ask_ai = tk.Button(button_frame, text="Ask AI", command=ask_ai_question, width=15, height=2)
btn_ask_ai.pack(side=tk.LEFT, padx=10)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

root.mainloop()
