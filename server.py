import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans

# Load the trained model and scaler
model = load_model('model.h5')
scaler = pickle.load(open('scaler1.pkl', 'rb'))

# Custom CSS for styling and hover effects, including background image
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    body {
        background-image: url('a1.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        font-family: 'Roboto', sans-serif;
        color: #333333;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #1B5E20;  /* Dark green color */
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stFileUploader label {
        display: flex;
        justify-content: center;
        font-weight: bold;
        color: #2E7D32;
    }
    .stButton button {
        background-color: #2E7D32;
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 30px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stButton button:hover {
        background-color: #1B5E20;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stImage img {
        display: block;
        margin: auto;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stImage img:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .prediction-result {
        font-weight: bold;
        font-size: 28px;
        text-align: center;
        margin-top: 30px;
        transition: all 0.3s ease;
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(46, 125, 50, 0.1);
    }
    .probabilities {
        color: #1565C0;
        font-weight: bold;
        font-size: 20px;
        text-align: center;
        margin-top: 15px;
        transition: color 0.3s ease;
    }
    .horizontal-line {
        border-top: 2px solid #2E7D32;
        margin: 30px 0;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def extract_features(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract GLCM features
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    features = {}
    features['Contrast'] = graycoprops(glcm, 'contrast')[0, 0]
    features['Correlation'] = graycoprops(glcm, 'correlation')[0, 0]
    features['Energy'] = graycoprops(glcm, 'energy')[0, 0]
    features['Homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
    
    # Calculate other features
    features['Mean'] = np.mean(gray_image)
    features['Standard Deviation'] = np.std(gray_image)
    features['Entropy'] = -np.sum(glcm * np.log2(glcm + (glcm==0)))
    features['Variance'] = np.var(gray_image)
    
    # Calculate skewness and kurtosis
    mean = np.mean(gray_image)
    features['Skewness'] = np.mean(((gray_image - mean) / np.std(gray_image)) ** 3)
    features['Kurtosis'] = np.mean(((gray_image - mean) / np.std(gray_image)) ** 4) - 3

    return features

def process_and_classify_image(image):
    # Save the uploaded file temporarily
    image_path = "temp.jpg"
    image.save(image_path)

    # Extract features
    features = extract_features(image_path)
    
    # Prepare feature vector
    feature_vector = np.array([[
        features['Contrast'], features['Correlation'], features['Energy'],
        features['Homogeneity'], features['Mean'], features['Standard Deviation'],
        features['Entropy'], features['Variance'], features['Kurtosis'],
        features['Skewness']
    ]])
    
    # Scale features
    scaled_features = scaler.transform(feature_vector)
    
    # Make prediction
    prediction = model.predict(scaled_features)
    
    threshold = 0.5
    if prediction[0][1] >= threshold:
        result = "Prediction: Bruised 🥺"
        color = "red"
    elif 0.3010 <= prediction[0][1] <= 0.5046:
        result = "Prediction: Moderate Bruised 😕"
        color = "orange"
    else:
        result = "Prediction: Non-Bruised 😊"
        color = "green"
        
    return prediction[0][0], prediction[0][1], result, color

def cluster_image(image_path, n_clusters=3):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_values)
    
    # Get the labels and cluster centers
    labels = kmeans.labels_
    centers = np.uint8(kmeans.cluster_centers_)

    # Convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

# Title with dark green color and centered
st.markdown("<h1>🍎 Apple Bruise Classifier</h1>", unsafe_allow_html=True)

st.write("Upload an image of an apple to classify:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Original and Segmented Images:")
    
    # Save uploaded image
    image_path = "temp.jpg"
    image.save(image_path)

    # Get clustered images
    cluster_img1 = cluster_image(image_path, 2)
    cluster_img2 = cluster_image(image_path, 3)
    cluster_img3 = cluster_image(image_path, 4)

    # Convert clustered images to Image objects
    cluster_img1 = Image.fromarray(cluster_img1)
    cluster_img2 = Image.fromarray(cluster_img2)
    cluster_img3 = Image.fromarray(cluster_img3)

    # Display images side by side
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(image, caption='Original Image', use_column_width=True)

    with col2:
        st.image(cluster_img1, caption='Segmented image 1', use_column_width=True)

    with col3:
        st.image(cluster_img2, caption='Segmented image  2', use_column_width=True)

    with col4:
        st.image(cluster_img3, caption='Segmented image ` 3', use_column_width=True)
    
    st.write('<hr class="horizontal-line">', unsafe_allow_html=True)
    
    with st.spinner("Classifying..."):
        # Classify the image
        non_bruised_prob, bruised_prob, result, color = process_and_classify_image(image)

    # Display results with emojis
    st.markdown(f'<p class="prediction-result" style="color:{color};">{result}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="probabilities">Predicted probabilities: Non-Bruised={non_bruised_prob:.4f}, Bruised={bruised_prob:.4f}</p>', unsafe_allow_html=True)
    
    st.write('<hr class="horizontal-line">', unsafe_allow_html=True)