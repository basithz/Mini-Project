import os
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your pre-trained model
model = load_model('fruit_model_basi.h5')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('upload.html')

# Define a route for the upload page
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    img_path = os.path.join('static', file.filename)
    file.save(img_path)
    
    # Preprocess the image to match the model's expected input
    img = Image.open(img_path).resize((227, 227))  # Resize image to (227, 227)
    img_array = np.array(img) / 255.0  # Normalize the image (scale pixel values to [0, 1])
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 227, 227, 3)
    
    # Make predictions using the model
    predictions = model.predict(img_array)
    probability = predictions[0][0]  # Since it's a binary classification, the output is a single probability value
    
    # Determine the class label based on the probability
    if probability >= 0.5:
        result = 'Rotten'
    else:
        result = 'Fresh'

    return render_template('result.html', result=result, image_file=file.filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5001)
