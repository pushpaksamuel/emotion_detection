import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from recommendation import get_recommendations  # Import recommendation module

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model("model/my_model.keras")

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Error loading image.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    reshaped = resized.reshape(1, 48, 48, 1).astype("float32") / 255.0

    pred = model.predict(reshaped)[0]
    emotion = emotion_labels[np.argmax(pred)]

    print(f"Predicted Emotion: {emotion}")

    # Get recommendations based on detected emotion
    recommendations = get_recommendations(emotion)
    if recommendations:
        print(f"Quote: {recommendations['quote']}")
        print(f"Song: {recommendations['song']}")
        print(f"Activity: {recommendations['activity']}")
        print(f"Music Link: {recommendations['music_link']}")

    # Display the image and emotion
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Emotion: {emotion}")
    plt.axis("off")
    plt.show()
