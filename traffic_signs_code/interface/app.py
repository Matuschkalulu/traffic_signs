import streamlit as st
import requests
from PIL import Image

# for the Streamlit interface
st.title("Traffic Sign Recognition")
st.write("Identifie traffic signs in images.")

image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if image is not None:
    # displaying the uploaded image
    contai= st.container()
    col1, col2, col3 = contai.columns([1,80,1])
    col2.image(Image.open(image), caption="Uploaded Image", use_column_width=True)

    # making a prediction
    if st.button("Classify"):
        # Prepare the image data

        files = {"file": image.getvalue()}
        # sending a request to API
        response = requests.post("http://127.0.0.1:8080/ImagePrediction", files=files)

        if response.status_code == 200:
            # the prediction result
            prediction = response.json()
            if prediction['Value'] >= 0.4:
                col2.error('This is a unreadable!', icon="🚨")
                print(prediction)
            if prediction['Value'] < 0.4:
                col2.success('This is a readable!', icon="✅")



        else:
            st.error("Failed to classify the image. Please try again.")
