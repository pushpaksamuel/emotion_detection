import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv("fer2013.csv")

# Preprocess the data
def preprocess(data):
    pixels = data['pixels'].tolist()
    images = np.array([np.fromstring(pixel, sep=' ').reshape(48, 48, 1) for pixel in pixels])
    images = images / 255.0
    labels = to_categorical(data['emotion'].values, num_classes=7)
    return images, labels

train_data = data[data['Usage'] == 'Training']
val_data = data[data['Usage'] == 'PublicTest']
test_data = data[data['Usage'] == 'PrivateTest']

X_train, y_train = preprocess(train_data)
X_val, y_val = preprocess(val_data)
X_test, y_test = preprocess(test_data)

# Build the model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.2f}")

# Save the model
model.save("model/my_model.keras")
print("✅ Model saved at model/my_model.keras")
model.save("emotion_model.h5")
print("✅ Model saved at emotion_model.h5")