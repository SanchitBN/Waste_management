import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = r"C:/Users/B N Sudanva/OneDrive/Desktop/archive/waste_classifier.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = load_model(model_path)

# Define class names (Ensure it matches training order)
class_names = ['glass', 'metal', 'paper', 'plastic']  # Update this if needed

# Provide the path to a single image for testing
img_path = r"C:/Users/B N Sudanva/OneDrive/Desktop/archive/garbage-dataset/metal/metal_181.jpg"  # Change this to your image path

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Test image not found: {img_path}")

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input size
img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the class
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)  # Get index of highest probability
predicted_class = class_names[predicted_index]
confidence = np.max(predictions)  # Get confidence score

# Print the result
print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")

# Display the image with the prediction
plt.imshow(img)
plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
plt.axis("off")
plt.show()
