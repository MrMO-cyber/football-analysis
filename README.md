# Football Analysis Project

This project analyzes football matches using state-of-the-art AI technologies to improve decision-making and provide insights into gameplay.

---

## **Features**
- **Object Detection**: Detects players, the ball, and other key elements on the field.
- **Decision Analysis**: Evaluates player decisions to suggest better alternatives (e.g., dribble, pass, or shoot).
- **Annotated Output**: Produces an annotated video with visualized AI insights.
- **Real-Time or Recorded Analysis**: Processes real-time games or pre-recorded match footage.

---

## **How It Works**

1. **Data Collection**:
   - Input a football match video or live footage.

2. **Object Detection**:
   - The system uses YOLOv8 for detecting players, the ball, and other objects in the video.

3. **Decision Analysis**:
   - AI evaluates player decisions using parameters like:
     - Distance and angle to the goal.
     - Nearby defenders or teammates.
     - Player positioning.
   - Suggestions for better gameplay decisions are generated.

4. **Video Output**:
   - The system creates an annotated video with insights for coaches and analysts.

---

## **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/MrMO-cyber/football-analysis.git
   cd football-analysis
