import gradio as gr
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_disease(img):
    # Preprocess the image
    img_array = image.img_to_array(img.resize((299, 299))) # Ensure target size matches model input
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    predictions = model.predict(img_array)[0] # Get the first (and only) prediction in the batch

    # Create a dictionary of predictions with class names
    predicted_probs = {index_to_class[i]: float(predictions[i]) for i in range(len(index_to_class))}

    return predicted_probs

# Create Gradio Interface
iface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5), # Show top 5 predictions
    title="Tomato Leaf Disease Detection",
    description="Upload an image of a tomato leaf to get a disease prediction."
)

# Launch the interface
iface.launch(debug=True)
