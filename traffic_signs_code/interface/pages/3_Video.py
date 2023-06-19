import streamlit as st
import requests
import os

# for the Streamlit interface
st.title("Traffic Sign Recognition")
st.write("Identifie traffic signs in videos.")

#logo
st.sidebar.image("https://static.vecteezy.com/system/resources/previews/009/458/871/original/traffic-signs-icon-logo-design-template-vector.jpg", width=100)

# center logo
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Loading CSS
url_css = os.path.join(os.getcwd(), 'traffic_signs_code','interface','frontend', 'css', 'streamlit.css')
local_css(url_css)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

# Loading CSS
    local_css("frontend/css/streamlit.css")
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "svi", "mkv"])

st.markdown('<style>...</style>', unsafe_allow_html=True)


if uploaded_file is not None:
    # displaying the uploaded image
    video_file = open(uploaded_file, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes, caption="Uploaded Video", use_column_width=True)

    # making a prediction
    if st.button("Classify"):
        # Prepare the image data
        video_data = uploaded_file.read()

        # sending a request to API
        recognition_url = " "
        files = {"video": video_data}
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
