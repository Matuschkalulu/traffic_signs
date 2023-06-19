import streamlit as st

st.title("Traffic Sign Recognition")
st.write("Project Information")

st.sidebar.image("https://static.vecteezy.com/system/resources/previews/002/388/506/non_2x/concept-design-with-traffic-signs-vector.jpg", width=250)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

# Loading CSS
local_css("frontend/css/streamlit.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

st.text_area('Purpose of project', '''
    info...
    ''')
