# ======================================================
# NATURAL IMAGES - TKINTER GUI APP
# Based on Natural Image Classifier Project
# ======================================================

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model

# Load your best trained model (make sure this file exists)
MODEL_PATH = r""
model = load_model(MODEL_PATH)

# Define emotion classes and image size
CATEGORIES = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']
IMG_SIZE = 150  # Match the input size used in training

# Function to predict emotion from uploaded image
def predict_image():
    global img_label, result_label

    # Open file dialog to select image
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Load and preprocess image
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            img_input = np.expand_dims(img_resized, axis=0).astype(np.float32)

            # Make prediction
            prediction = model.predict(img_input)
            predicted_class_index = np.argmax(prediction)
            predicted_class = CATEGORIES[predicted_class_index]
            confidence = prediction[0][predicted_class_index] * 100

            # Display image
            img_pil = Image.fromarray((img_resized * 255).astype(np.uint8))
            img_pil = img_pil.resize((256, 256))  # Resize for display
            img_tk = ImageTk.PhotoImage(img_pil)
            img_label.config(image=img_tk)
            img_label.image = img_tk  # Keep reference to avoid garbage collection

            # Show result
            result_label.config(text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")

        except Exception as e:
            result_label.config(text=f"Error: {str(e)}")
    else:
        result_label.config(text="No image selected.")

# Create main window
root = tk.Tk()
root.title("Emotion Recognition GUI")
root.geometry("400x550")
root.resizable(False, False)
root.configure(bg="#f0f0f0")

# Title label
title_label = tk.Label(root, text="Natural Image Classifier", font=("Helvetica", 18, "bold"), bg="#f0f0f0")
title_label.pack(pady=10)

# Upload button
upload_btn = tk.Button(
    root,
    text="Upload Image",
    command=predict_image,
    font=("Helvetica", 12),
    bg="#4CAF50",
    fg="white",
    padx=20,
    pady=10
)
upload_btn.pack(pady=10)

# Image display area
img_label = tk.Label(root)
img_label.pack()

# Result label
result_label = tk.Label(
    root,
    text="No image selected.",
    font=("Helvetica", 14),
    justify="center",
    bg="#f0f0f0",
    wraplength=300
)
result_label.pack(pady=20)

# Start GUI loop
root.mainloop()