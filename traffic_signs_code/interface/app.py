import streamlit as st
import requests
from PIL import Image

# for the Streamlit interface
st.title("Traffic Sign Recognition")
st.write("Identifie traffic signs in images.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # displaying the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # making a prediction
    if st.button("Classify"):
        # Prepare the image data
        image_data = uploaded_file.read()

        # sending a request to API
        recognition_url = " "
        files = {"image": image_data}
        response = requests.post(recognition_url, files=files)

        if response.status_code == 200:
            # the prediction result
            prediction = response.json()
            traffic_sign = prediction["traffic_sign"]
            confidence = prediction["confidence"]

            st.success(f"Predicted traffic sign: {traffic_sign}")
            st.info(f"Confidence: {confidence}")
        else:
            st.error("Failed to classify the image. Please try again.")
