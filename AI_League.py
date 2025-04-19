import cv2
import numpy as np
from ultralytics import YOLO
from google import genai  # Import the Google GenAI client

# Initialize the GenAI client with your API key
client = genai.Client(api_key="")  # Replace with your actual Google GenAI API key

# Load YOLOv8 model for object detection
model = YOLO("yolov8n.pt")  # Replace with your desired YOLOv8 model weights (e.g., yolov8s.pt)

# Open video
video_path = r"C:\Users\ASUS\Downloads\torres_missed.mp4" # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Video output configuration
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output_analysis.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Configurable parameters
DEFENDER_RADIUS = 100  # Radius to consider a player as a "nearby defender"
GOAL_MARGIN_TOP = 50  # Goal position (distance from the top of the screen)

# Helper function to calculate the center of a bounding box
def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# Function to analyze player decisions
def analyze_decision(ball_pos, players, frame):
    if not players:
        return "No players detected."
    if not ball_pos:
        return "Ball not detected. Focusing on player positions."

    bx, by = ball_pos

    # Find the closest player to the ball
    closest_player = None
    min_distance = float('inf')

    for player in players:
        px, py = get_center(player)
        dist = np.linalg.norm(np.array([bx, by]) - np.array([px, py]))
        if dist < min_distance:
            min_distance = dist
            closest_player = player

    if closest_player:
        px, py = get_center(closest_player)

        # Calculate the goal position (top-center of the frame)
        goal_x, goal_y = frame.shape[1] // 2, GOAL_MARGIN_TOP

        # Count nearby defenders
        nearby_defenders = 0
        for other in players:
            if other == closest_player:
                continue
            ox, oy = get_center(other)
            if np.linalg.norm(np.array([px, py]) - np.array([ox, oy])) < DEFENDER_RADIUS:
                nearby_defenders += 1

        # Create a scenario description
        scenario = {
            "player_position": (px, py),
            "ball_position": (bx, by),
            "goal_position": (goal_x, goal_y),
            "nearby_defenders": nearby_defenders,
        }

        # Generate correction using Gemini
        return generate_correction(scenario)

    return "No player close to the ball."

# Function to generate a correction using Gemini
def generate_correction(scenario):
    prompt = (
        f"A football player is in position {scenario['player_position']} with the ball at {scenario['ball_position']}. "
        f"The goal is located at {scenario['goal_position']} and there are {scenario['nearby_defenders']} defenders nearby. "
        f"Analyze the player's current decision and suggest a better alternative if the current decision is wrong."
    )

    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating correction: {e}"

# Main loop to process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection with YOLOv8
    results = model(frame, conf=0.25)  # Lower confidence threshold for better detection

    ball_pos = None
    players = []

    # Process detection results
    for result in results:
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()[:4]
            cls = int(box.cls[0].item())  # Get class index
            label = model.names[cls]  # Get class label

            if label == "person":
                # Add player bounding box
                players.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, "Player", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            elif label == "sports ball":
                # Detect ball and calculate its center
                ball_pos = get_center((x1, y1, x2, y2))
                cv2.circle(frame, ball_pos, 8, (0, 255, 0), -1)
                cv2.putText(frame, "Ball", (ball_pos[0] + 10, ball_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Analyze the player's decision
    correction = analyze_decision(ball_pos, players, frame)
    print(f"Correction: {correction}")

    # Annotate the suggestion on the frame
    cv2.putText(frame, correction, (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show the frame and save it to the output video
    cv2.imshow("Football Match Analysis", frame)
    out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
