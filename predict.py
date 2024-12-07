import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load the trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    
# Reverse the class indices dictionary to map indices to class names
class_names = {v: k for k, v in class_indices.items()}

def preprocess_image(image_path):
    """Preprocess the input image for prediction."""
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match training input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = img_array / 255.  # Rescale to match training
    return img_array

def predict_disease(image_path):
    """Predict the disease from an image."""
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # Get the class name
    predicted_class = class_names[predicted_class_index]
    
    return predicted_class, confidence

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    try:
        disease, confidence = predict_disease(image_path)
        print(f"Predicted Disease: {disease}")
        print(f"Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
