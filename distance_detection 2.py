import cv2
import winsound
import time
import matplotlib.pyplot as plt

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

KNOWN_WIDTH = 14
FOCAL_LENGTH = 500

cap = cv2.VideoCapture(0)

last_beep = 0
distance_values = []

# For smoothing
distance_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 6)

    for (x, y, w, h) in faces:
        # Raw distance
        distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w

        # Smooth distance
        distance_buffer.append(distance)
        if len(distance_buffer) > 5:
            distance_buffer.pop(0)

        distance = sum(distance_buffer) / len(distance_buffer)

        # Store for graph
        distance_values.append(distance)
        if len(distance_values) > 100:
            distance_values.pop(0)

        # WARNING + SOUND
        if distance < 40:
            cv2.putText(frame, "WARNING: Too Close!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if time.time() - last_beep > 1:
                winsound.Beep(1000, 300)
                last_beep = time.time()

        # Color based on distance
        if distance < 40:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Show distance
        cv2.putText(frame, f"Distance: {int(distance)} cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Distance Detection System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# 📊 GRAPH AFTER CLOSING CAMERA
plt.plot(distance_values)
plt.title("Distance vs Time")
plt.xlabel("Frames")
plt.ylabel("Distance (cm)")
plt.grid()
plt.show()