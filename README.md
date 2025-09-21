# Pneumonia-Detection-from-Chest-X-Rays-using-CNN
Pneumonia Detection System is a deep learning-based web app that classifies chest X-ray images as Pneumonia or Normal using a CNN model. Built with Python, TensorFlow, and Flask/Streamlit, it provides an image upload interface with Grad-CAM heatmaps for accurate and explainable predictions.
# 🫁 Pneumonia Detection System

A deep learning-based web application that detects **Pneumonia** from chest X-ray images using a **Convolutional Neural Network (CNN)**. Users can upload X-ray images and get predictions along with **Grad-CAM heatmaps** to visualize which regions influenced the model's decision.  

---

## 🚀 Features
- Upload chest X-ray images through a simple web interface  
- Classify images as **Pneumonia** or **Normal**  
- Visualize Grad-CAM heatmaps for explainable predictions  
- Built with **Python, TensorFlow/Keras, and Flask/Streamlit**  

---

## 🛠️ Tech Stack
- Python 3  
- TensorFlow / Keras  
- Flask / Streamlit  
- OpenCV, NumPy, PIL for image processing  
- Matplotlib for visualizations  

---

## 📂 Project Structure
pneumonia-detection/
│── app.py # Main Flask/Streamlit app
│── model_utils.py # Model loading & prediction functions
│── pneumonia_detection_cnn_model.h5 # Trained CNN model
│── static/uploads/ # Uploaded images
│── templates/ # HTML files (if Flask)
│── requirements.txt # Python dependencies
│── README.md # Project documentation
Upgrade pip
python -m pip install --upgrade pip setuptools wheel

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application

For Streamlit:

streamlit run app.py


For Flask:

python app.py


Then open your browser at:

http://localhost:8501  # Streamlit default
http://127.0.0.1:5000 # Flask default

5️⃣ Upload an X-ray image

Click the upload button on the interface

The model will predict Pneumonia or Normal

Grad-CAM heatmap will show areas influencing the prediction

🧠 Model Training (Optional)

Dataset:https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Model: Custom CNN architecture trained on X-ray images

Performance: ~90%+ Accuracy
