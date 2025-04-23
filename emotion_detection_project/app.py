import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import time
import gdown
import os

# Create 'model' folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Correct path and Google Drive direct link
model_path = "model/my_model.keras"
url = "https://drive.google.com/uc?id=1-JxgNAvYSCstSbIXJYMmPIMPgKS30Ibu"

# Download model if not present
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Check if file is downloaded correctly
if os.path.exists(model_path):
    print("Downloaded file size:", os.path.getsize(model_path))
    print("File exists:", os.path.exists(model_path))
    print("File size:", os.path.getsize(model_path))
    model = load_model(model_path)
else:
    raise FileNotFoundError("Model file could not be downloaded correctly.")


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emotion-based recommendations
emotion_recommendations = {
    "Angry": {
        "quote": "For every minute you are angry you lose sixty seconds of happiness. - Ralph Waldo Emerson",
        "song": "Imagine Dragons - Believer",
        "activity": "Take deep breaths and try a quick walk outside to cool off.",
        "music_link": "https://www.youtube.com/watch?v=7wtfhZwyrcc"
    },
    "Disgust": {
        "quote": "Disgust is the mother of all moral emotions. - Peter Singer",
        "song": "Coldplay - Fix You",
        "activity": "Reflect on what caused the disgust and try to change your environment.",
        "music_link": "https://www.youtube.com/watch?v=YQHsXMglC9A"
    },
    "Fear": {
        "quote": "The only thing we have to fear is fear itself. - Franklin D. Roosevelt",
        "song": "Lindsey Stirling - Crystallize",
        "activity": "Take a few moments to breathe and think of positive outcomes.",
        "music_link": "https://www.youtube.com/watch?v=aHjpOzsQ9YI"
    },
    "Happy": {
        "quote": "Happiness depends upon ourselves. - Aristotle",
        "song": "Pharrell Williams - Happy",
        "activity": "Share your happiness with others, spread positivity.",
        "music_link": "https://www.youtube.com/watch?v=ZbZSe6N_BXs"
    },
    "Sad": {
        "quote": "Tears are words that need to be written. - Paulo Coelho",
        "song": "Adele - Someone Like You",
        "activity": "Talk to a friend or take some time to reflect and relax.",
        "music_link": "https://www.youtube.com/watch?v=hLQl3WQQoQ0"
    },
    "Surprise": {
        "quote": "Life is full of surprises, but you need to be open to them. - Unknown",
        "song": "Queen - Don't Stop Me Now",
        "activity": "Embrace the surprise and go with the flow.",
        "music_link": "https://www.youtube.com/watch?v=2xW3WsS0F6g"
    },
    "Neutral": {
        "quote": "Just because you don’t feel anything, doesn’t mean you’re numb. - Unknown",
        "song": "Lofi Hip Hop for studying",
        "activity": "Relax, take a break, or enjoy something creative.",
        "music_link": "https://www.youtube.com/watch?v=5qap5aO4i9A"
    }
}

# Emotion prediction function
def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.reshape(1, 48, 48, 1) / 255.0
    preds = model.predict(face)
    return emotion_labels[np.argmax(preds)]

# Streamlit UI
st.set_page_config(page_title="Real-Time Emotion Detection", layout="centered", page_icon="🧠")
st.title("🧠 Real-Time Emotion Detection")

input_type = st.radio("Choose input type:", ["Image 🖼️", "Video 📹"])

if input_type == "Image 🖼️":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert PIL image to OpenCV format
        image_np = np.array(image.convert('RGB'))
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Predict emotion
        emotion = predict_emotion(image_cv)
        st.success(f"Detected Emotion: **{emotion}**")

        # Recommendations
        recommendations = emotion_recommendations.get(emotion, None)
        if recommendations:
            st.write(f"💬 Quote: {recommendations['quote']}")
            st.write(f"🎵 Song: {recommendations['song']}")
            st.write(f"▶️ Music Link: {recommendations['music_link']}")
            st.write(f"🧘 Activity: {recommendations['activity']}")

elif input_type == "Video 📹":
    st.info("📸 Click the button below to start webcam (20 sec)")
    start_camera = st.button("Start Webcam")

    if start_camera:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        start_time = time.time()
        final_emotion = None

        while time.time() - start_time < 20:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to access the camera.")
                break

            # Flip and predict
            frame = cv2.flip(frame, 1)
            try:
                emotion = predict_emotion(frame)
                final_emotion = emotion
                cv2.putText(frame, f"Emotion: {emotion}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except:
                cv2.putText(frame, "Face not detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            stframe.image(frame, channels="BGR")

        cap.release()
        st.success("✅ Detection session ended.")

        if final_emotion:
            st.write(f"Detected Final Emotion: **{final_emotion}**")
            recommendations = emotion_recommendations.get(final_emotion, None)
            if recommendations:
                st.write(f"💬 Quote: {recommendations['quote']}")
                st.write(f"🎵 Song: {recommendations['song']}")
                st.write(f"▶️ Music Link: {recommendations['music_link']}")
                st.write(f"🧘 Activity: {recommendations['activity']}")
