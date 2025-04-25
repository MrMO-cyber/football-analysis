# Football Analysis Project
Overview
This program is a video analysis tool built with Python, utilizing the YOLOv8 object detection model and an AI-powered question-answering system. The application provides the following functionalities:

Video Analysis: Detects objects (e.g., players, ball) in a football video and logs events.
AI Interaction: Lets users ask questions about the video, and the AI generates concise answers based on the detected events.
Graphical User Interface (GUI): A user-friendly interface built with Tkinter for video selection, processing, and interaction.
Key Features
YOLOv8 Object Detection:

Detects objects like players and the ball in a football match video.
Highlights detected objects with bounding boxes and labels.
Identifies if a player is a goalkeeper based on their position in the penalty box.
Event Logging:

Logs events like ball and player detections.
Captures object locations, confidence levels, and dominant colors of objects.
AI-Powered Summary:

Allows users to ask natural language questions about the match (e.g., "How many times was the ball in the penalty area?").
Utilizes Google Generative AI (gemini-1.5-flash) to generate answers based on logged events.
Graphical User Interface (GUI):

Video selection and playback.
Start/stop video analysis.
Progress bar to indicate analysis progress.
Display of detection results and AI-generated summaries.
Requirements
Libraries Used
Object Detection:

ultralytics: For YOLOv8 model.
opencv-python: For video processing and drawing bounding boxes.
AI Integration:

google-generativeai: For generating natural language answers to user queries.
GUI:

tkinter: For the graphical user interface.
Pillow: For rendering video frames in the GUI.
Miscellaneous:

numpy: For array manipulations.
collections.Counter: For calculating dominant object colors.
Installation
Install the required Python libraries using the following commands:
"pip install ultralytics opencv-python google-generativeai Pillow numpy"
Usage Instructions
Run the Program:
Execute the script:
"python script.py"
Select a Video:
Click the "Browse Video" button to select a football match video.
Supported formats: .mp4, .avi, .mov, .mkv.
Start Analysis:

Click the "Start Analysis & Summary" button.
The program will:
Detect objects in the video.
Highlight objects and log events.
Save the analyzed video as output_analysis.mp4.
Ask Questions:

After analysis, click the "Ask AI" button.
Enter a question about the match (e.g., "Where was the ball most frequently detected?").
The AI will generate and display an answer based on the detected events.
View Logs and Summaries:

The program provides a summary of detected events and allows you to view detailed logs.
Code Structure
1. Configuration
Sets up the Google Generative AI API key.
Loads the YOLOv8 model (yolov8l.pt).
2. Core Functionalities
Object Detection:
get_center(box): Calculates the center of a bounding box.
is_goalkeeper(box, frame_width, frame_height): Determines if a player is in the penalty box.
get_dominant_color(frame, box): Identifies the dominant color of an object.
Video Analysis:
analyze_video(): Processes the video frame by frame:
Detects objects using YOLOv8.
Logs detected events.
Displays frames in the GUI.
AI Question Answering:
ask_ai_question(): Prompts the user for a question.
Sends a summary of logged events to the AI model.
Displays the AI-generated answer.
3. Graphical User Interface (GUI)
Built using tkinter.
Includes buttons for video selection, analysis, and AI interaction.
Displays video frames and progress during analysis.
Notes
Google API Key:

Replace the placeholder API key (GOOGLE_API_KEY) with a valid key.
YOLO Model:

Ensure the pre-trained YOLOv8 weights file (yolov8l.pt) is available.
Performance:

The program skips frames (skip_frames = 1) to balance speed and accuracy.
Error Handling:

The program includes error messages for common issues (e.g., invalid video files, missing API key).
Future Enhancements
Add support for real-time video analysis via webcam.
Integrate advanced AI models for more detailed event recognition.
Optimize performance for longer videos by parallelizing frame processing.

This program provides a comprehensive platform for football video analysis, combining state-of-the-art object detection with AI-driven insights.
