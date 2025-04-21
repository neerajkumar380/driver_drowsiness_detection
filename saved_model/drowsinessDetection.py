import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model

# Initialize pygame for sound
pygame.mixer.init()
alarm_sound = "alarm.wav"

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(alarm_sound)
        pygame.mixer.music.play()

# Load the trained model
model = load_model("saved_model/drowsiness_model.h5")

# Load Haar cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Open webcam
cap = cv2.VideoCapture(0)

CATEGORIES = ["Closed_Eyes", "Open_Eyes"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    predictions = []

    for (x, y, w, h) in eyes[:2]:  # Only first two eyes detected (left + right)
        eye = gray[y:y+h, x:x+w]
        eye_resized = cv2.resize(eye, (64, 64)).reshape(-1, 64, 64, 1) / 255.0
        prediction = model.predict(eye_resized, verbose=0)
        predictions.append(np.argmax(prediction))

        # Draw rectangle around eye
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    if predictions:
        # Majority voting if both eyes detected
        final_label = max(set(predictions), key=predictions.count)
        label_text = CATEGORIES[final_label]

        # Show label
        cv2.putText(frame, label_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Alarm if closed eyes
        if label_text == "Closed_Eyes":
            print("⚠️ ALERT! Driver is Drowsy! ⚠️")
            play_alarm()
    else:
        cv2.putText(frame, "No Eyes Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
pygame.mixer.music.stop()
pygame.quit()
cap.release()
cv2.destroyAllWindows()
