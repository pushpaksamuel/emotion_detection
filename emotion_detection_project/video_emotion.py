import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
from recommendation import get_recommendations  # your separate recommendation module

# Emotion labels and model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model("model/my_model.keras")

def process_video():
    print("[INFO] Initializing webcam...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot access the webcam.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    start_time = time.time()
    duration = 20
    last_emotion = None

    print("[INFO] Running emotion detection for 20 seconds... Press 'q' to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Couldn't grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48)).astype("float32") / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))

            pred = model.predict(roi, verbose=0)[0]
            emotion = emotion_labels[np.argmax(pred)]
            last_emotion = emotion  # update last seen emotion

            # draw live rectangle + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(
                frame, emotion, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )

        cv2.imshow("🎥 Video Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or (time.time() - start_time) > duration:
            break

    cap.release()
    cv2.destroyAllWindows()

    # After video ends, show recommendation
    if last_emotion:
        rec = get_recommendations(last_emotion)
        if rec:
            print(f"\n✅ Final Detected Emotion: {last_emotion}")
            print(f"💬 Quote: {rec['quote']}")
            print(f"🎵 Song: {rec['song']}")
            print(f"▶️ Music Link: {rec['music_link']}")
            print(f"🧘 Activity: {rec['activity']}")
        else:
            print("\n[INFO] No recommendation found for:", last_emotion)
    else:
        print("\n[INFO] No face/emotion detected during the session.")
