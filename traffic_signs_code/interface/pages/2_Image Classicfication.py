import streamlit as st
import requests

st.title("Traffic Sign Recognition")
st.write("Classicfication on images")

st.sidebar.image("https://static.vecteezy.com/system/resources/previews/009/458/871/original/traffic-signs-icon-logo-design-template-vector.jpg", width=100)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

# Loading CSS
url_css = os.path.join(os.getcwd(), 'traffic_signs_code','interface','frontend', 'css', 'streamlit.css')
local_css(url_css)
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
