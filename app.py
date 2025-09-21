from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model/pneumonia_detection_cnn_model.h5")

def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]  # Raw probability
    percentage = round(prediction * 100, 2)  # Convert to percentage

    label = "🫁 Pneumonia Detected" if prediction > 0.5 else "✅ Normal"
    return label, percentage

@app.route('/')
def index():
    return render_template('index.html', prediction=None, percentage=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    result, percentage = predict_pneumonia(filepath)
    os.remove(filepath)
    return render_template('index.html', prediction=result, percentage=percentage)

if __name__ == '__main__':
    app.run(debug=True)
