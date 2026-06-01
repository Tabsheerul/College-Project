import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the Dataset Spreadsheet
print("Loading the dataset...")
data = pd.read_csv("gesture_dataset.csv")

# 2. Split into Features (Math) and Labels (Letters)
X = data.drop('label', axis=1).values # The 42 math coordinates
y_text = data['label'].values         # The actual letters (A, B, C)

# 3. Convert Letters to Numbers (A=0, B=1, C=2)
encoder = LabelEncoder()
y = encoder.fit_transform(y_text)

classes = encoder.classes_
print(f"The AI is learning these gestures: {classes}")

# 4. Split the data (80% for studying, 20% for taking a test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- NEW: Reshape Data for LSTM ---
# LSTM networks require 3D input: (number_of_samples, time_steps, features)
# Since our CSV has static frames, our time_steps = 1
X_train_lstm = np.expand_dims(X_train, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)

# 5. Build the Brain (LSTM Architecture)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=False, input_shape=(1, X_train.shape[1])),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax') 
])

# 6. Compile the Brain
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 7. Train the Brain!
print("Training the AI...")
history = model.fit(X_train_lstm, y_train, epochs=50, validation_data=(X_test_lstm, y_test))

# 8. Save the Brain
model.save("upgraded_model.h5")
print("\nTraining Complete! Brain saved successfully as 'upgraded_model.h5'")
print(f"*** IMPORTANT: Remember this exact order for the final step: {list(classes)} ***")