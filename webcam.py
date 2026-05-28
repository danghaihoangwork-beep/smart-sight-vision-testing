import cv2
from ultralytics import YOLO

# 1. Load your AI "brain" into the system
model = YOLO('best.pt')

# 2. Start the Webcam (0 is usually the default laptop camera)
cap = cv2.VideoCapture(0)

print("🚀 Starting SmartSight... Press 'q' to exit the program.")
print("💡 MANUAL: Index finger = LEFT | Thumb = RIGHT")

while cap.isOpened():
    # Read each frame from the Webcam
    success, frame = cap.read()
    if not success:
        print("❌ Cannot connect to the Webcam!")
        break

    # Flip the image horizontally to act like a mirror (Makes it easier to control your hand)
    frame = cv2.flip(frame, 1)

    # 3. Pass the frame to the AI for prediction
    # conf=0.5: Only detect when the AI is over 50% confident
    # verbose=False: Turn off continuous terminal logging to keep it clean
    results = model.predict(frame, conf=0.5, verbose=False)

    # 4. The AI automatically draws bounding boxes and text on the frame
    annotated_frame = results[0].plot()

    # 5. Display the Video window on the screen
    cv2.imshow("SmartSight - Hand Direction Tracker", annotated_frame)

    # 6. Listen for keyboard input, break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up the camera and close all windows after exiting
cap.release()
cv2.destroyAllWindows()