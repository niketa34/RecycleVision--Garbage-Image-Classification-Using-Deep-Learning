import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load best model (download from Google Drive to local folder first)
model = tf.keras.models.load_model("garbage_classifier_best.h5")

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Garbage Classification"])

if page == "Introduction":
    st.title("♻️ Garbage Classification Project")
    st.write("""
    This project uses deep learning to classify waste into 6 categories:
    cardboard, glass, metal, paper, plastic, and trash.
    Upload an image in the Garbage Classification page to see predictions.
    """)

elif page == "Garbage Classification":
    st.title("Garbage Classification App")
    uploaded_file = st.file_uploader("Upload a garbage image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img = image.resize((224, 224))
        img_array = np.array(img) / 127.5 - 1
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        top_indices = preds[0].argsort()[-3:][::-1]

        st.subheader("Prediction Results:")
        for i in top_indices:
            st.write(f"{class_names[i]}: {preds[0][i]*100:.2f}%")