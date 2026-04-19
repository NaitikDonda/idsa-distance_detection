import cv2
import os
import platform
import time
import threading
import base64
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt

def beep():
    """Cross-platform beep sound - non-blocking"""
    def _play_sound():
        system = platform.system()
        if system == "Windows":
            import winsound
            winsound.Beep(1000, 300)
        elif system == "Darwin":  # macOS
            os.system('afplay /System/Library/Sounds/Ping.aiff')
        else:  # Linux
            print('\a')  # Terminal bell
    
    # Run sound in separate thread to avoid lag
    threading.Thread(target=_play_sound, daemon=True).start()

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

KNOWN_WIDTH = 14
FOCAL_LENGTH = 500

# Use lower resolution for better performance
PROCESS_WIDTH = 320
PROCESS_HEIGHT = 240

# Process every Nth frame (skip frames)
SKIP_FRAMES = 2

# Efficient circular buffers using deque
distance_buffer = deque(maxlen=5)
distance_values = deque(maxlen=100)

last_distance = None
last_face_coords = None
last_frame = None
frame_count = 0

cap = cv2.VideoCapture(0)

# Set camera to lower resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_beep = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Skip face detection on some frames for performance
    if frame_count % SKIP_FRAMES == 0:
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) > 0:
            # Get the largest face (closest to camera)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]

            # Scale coordinates back to original frame size
            scale_x = frame.shape[1] / PROCESS_WIDTH
            scale_y = frame.shape[0] / PROCESS_HEIGHT
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

            # Raw distance
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w

            # Smooth distance using efficient deque
            distance_buffer.append(distance)
            distance = sum(distance_buffer) / len(distance_buffer)
            last_distance = distance
            last_face_coords = (x, y, w, h)

            # Store for graph
            distance_values.append(distance)

    # Use last known distance if no face detected this frame
    distance = last_distance if last_distance else 100

    # WARNING + SOUND
    if distance < 40:
        cv2.putText(frame, "WARNING: Too Close!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if time.time() - last_beep > 1:
            beep()
            last_beep = time.time()

    # Color based on distance
    if distance < 40:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)

    # Draw rectangle
    if last_face_coords:
        x, y, w, h = last_face_coords
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Show distance
        cv2.putText(frame, f"Distance: {int(distance)} cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Store last frame for snapshot
    last_frame = frame.copy()

    cv2.imshow("Distance Detection System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    # Check if window was closed
    if cv2.getWindowProperty("Distance Detection System", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

# Create Jupyter notebook with snapshot and graph
def create_notebook():
    import json
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    # Save to Downloads folder
    save_dir = os.path.expanduser("~/Downloads")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    notebook_path = os.path.join(save_dir, f"distance_detection_report_{timestamp}.ipynb")

    cells = []

    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# Distance Detection Report\n", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    })

    # Snapshot cell
    if last_frame is not None:
        # Convert frame to PNG bytes
        _, buffer = cv2.imencode('.png', last_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Last Snapshot"]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [{
                "output_type": "display_data",
                "data": {
                    "image/png": img_base64,
                    "text/plain": ["<IPython.core.display.Image object>"]
                },
                "metadata": {}
            }],
            "source": ["from IPython.display import Image, display\n", f"display(Image(data=base64.b64decode('{img_base64}')))"]
        })

    # Graph cell
    if distance_values:
        # Create graph and save to buffer
        plt.figure(figsize=(10, 5))
        plt.plot(list(distance_values))
        plt.title("Distance vs Time")
        plt.xlabel("Frames")
        plt.ylabel("Distance (cm)")
        plt.grid()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Distance Graph"]
        })
        cells.append({
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [{
                "output_type": "display_data",
                "data": {
                    "image/png": graph_base64,
                    "text/plain": ["<matplotlib.figure.Figure>"]
                },
                "metadata": {}
            }],
            "source": ["import matplotlib.pyplot as plt\n", "from io import BytesIO\n", "import base64\n", "plt.figure(figsize=(10, 5))\n", f"plt.plot({list(distance_values)})\n", "plt.title('Distance vs Time')\n", "plt.xlabel('Frames')\n", "plt.ylabel('Distance (cm)')\n", "plt.grid()\n", "plt.show()"]
        })

    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Save notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)

    print(f"Report saved: {notebook_path}")
    return notebook_path

# Generate the notebook
create_notebook()