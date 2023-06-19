import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("Traffic Sign Recognition")
st.write("Detecting on images")

st.sidebar.image("https://static.vecteezy.com/system/resources/previews/002/388/506/non_2x/concept-design-with-traffic-signs-vector.jpg", width=250)


def remote_css(url):
 st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

# Loading CSS
local_css("frontend/css/streamlit.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

import requests


def detect_same_image(api_url, local_image_path):
    # loading the local image
    with open(local_image_path, 'rb') as image_file:
        local_image = Image.open(image_file)
        local_image_data = local_image.tobytes()

    # request for the API
    response = requests.get(api_url)
    response.raise_for_status()

    # loading the API response image
    api_image = Image.open(BytesIO(response.content))
    api_image_data = api_image.tobytes()

    # comparing image data
    if local_image_data == api_image_data:
        print("The image is reganise.")
    else:
        print("The image is unrecganise.")

# usage
api_url = " "
local_image_path = "path/to/local/image.jpg"
detect_same_image(api_url, local_image_path)
