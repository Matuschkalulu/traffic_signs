import streamlit as st
import requests
from PIL import Image

# for the Streamlit interface
st.title("Traffic Sign Recognition")
st.write("Identifie traffic signs in images.")

st.sidebar.image("https://static.vecteezy.com/system/resources/previews/002/388/506/non_2x/concept-design-with-traffic-signs-vector.jpg", width=250)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Loading CSS
local_css("frontend/css/streamlit.css")


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

# Loading CSS
    local_css("frontend/css/streamlit.css")
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

st.markdown('<style>...</style>', unsafe_allow_html=True)

if uploaded_file is not None:
    # displaying the uploaded image
    image = Image.open(uploaded_file)

    st.header("Traffic sign that you want")

    with st.columns(1)[0]:
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
            # the predictation result
            prediction = response.json()
            traffic_sign = prediction["traffic_sign"]
            confidence = prediction["confidence"]

            st.success(f"Predicted traffic sign: {traffic_sign}")
            st.info(f"Confidence: {confidence}")
        else:
            st.error("Failed to classify the image. Please try again.")
