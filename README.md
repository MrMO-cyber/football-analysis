# Football Analysis Project

This project uses **YOLOv8** for object detection and **Google GenAI** (Gemini) for decision-making analysis in football matches. It processes football match footage, detects players and the ball, evaluates player decisions, and provides AI-generated suggestions for better gameplay strategies.

---

## **How the Code Works**

### **1. Objectives**
- Detect players and the ball in a football match video.
- Analyze player decisions (e.g., pass, dribble, shoot) using AI.
- Provide suggestions for better gameplay strategies using Google's Generative AI.

---

### **2. Code Breakdown**

#### **Step 1: Import Libraries**
The script imports the required libraries:
- `cv2`: For video processing and frame display.
- `numpy`: For mathematical operations like calculating distances.
- `ultralytics`: For loading and using the YOLOv8 object detection model.
- `google.genai`: For accessing Google's Generative AI services.

#### **Step 2: Initialize Models**
- **Google GenAI Client**: Initialized with an API key to access the Generative AI model (`gemini-2.0-flash`).
- **YOLOv8 Model**: Pre-trained weights (`yolov8n.pt`) are loaded for object detection.

#### **Step 3: Open Input Video**
- The script opens the input video (`torres_missed.mp4`) using OpenCV.
- It also sets up the output video configuration to save the annotated video.

#### **Step 4: Helper Functions**
- **`get_center()`**: Calculates the center point of a bounding box.
- **`analyze_decision()`**: Analyzes the closest player's decision based on:
  - Ball position.
  - Player position.
  - Goal position.
  - Number of nearby defenders.
- **`generate_correction()`**: Sends a scenario description to the Google GenAI model and retrieves a suggested correction.

#### **Step 5: Main Video Processing**
- The script processes each frame of the video:
  - **Object Detection**: YOLOv8 detects players (labeled as "person") and the ball (labeled as "sports ball").
  - **Decision Analysis**:
    - Finds the closest player to the ball.
    - Analyzes their decision using the ball's position, player's position, and nearby defenders.
    - Calls the Generative AI model to suggest a better alternative, if needed.
  - **Annotations**:
    - Draws bounding boxes around detected players and the ball.
    - Displays the suggestion (correction) on the video frame.

#### **Step 6: Save and Display Results**
- Annotated video frames are saved to an output file (`output_analysis.mp4`).
- The processed video is displayed in real-time, and the loop can be stopped by pressing the `q` key.

---

### **3. Features**
- **Object Detection**:
  - YOLOv8 detects players and the ball in the video.
  - Bounding boxes and labels are drawn for visual identification.

- **Decision Analysis**:
  - Evaluates the closest player's decision based on:
    - Position of the player.
    - Ball's position.
    - Goal's position.
    - Number of nearby defenders.
  - Calls Google's Generative AI (Gemini) to provide suggestions for better decision-making.

- **Video Output**:
  - Annotates the video with detections and AI-generated suggestions.
  - Saves the annotated video to the specified output file.

---

### **4. Installation and Usage**

#### **Installation**
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
