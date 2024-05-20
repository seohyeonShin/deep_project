# this from 
#https://fulldataalchemist.medium.com/building-your-own-real-time-object-detection-app-roboflow-yolov8-and-streamlit-part-4-16a025c7240c

# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Sign Translation using CNNtoLSTM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection")
st.caption('Updload a photo with this :blue[hand signals]: :+1:, :hand:, :i_love_you_hand_sign:, and :spock-hand:.')
st.caption('Then click the :blue[Detect Objects] button and check the result.')

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
# model_type = st.sidebar.radio(
#     "Select Task", ['Detection', 'Segmentation'])

time_step = int(st.sidebar.slider(
    "Select TTS Model Timestep", 4, 1000, 50))

# Selecting Detection Or Segmentation

tts_model_path = Path(settings.TTS_MODEL)
mt_model_path = Path(settings.MT_MODEL)

# Load Pre-trained ML Model
try:
    tts_model = helper.load_model(tts_model_path, type='tts')
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {tts_model_path}")
    st.error(ex)

try:
    # MT model load
    mt_model=helper.load_model(mt_model_path, type='mt')
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {mt_model_path}")
    st.error(ex)

st.sidebar.header("Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
model = [tts_model, mt_model]

# if source_radio == settings.VIDEO:
#     helper.play_stored_video(time_step, model)

if source_radio == settings.WEBCAM:
    helper.play_webcam(time_step, model)

else:
    st.error("Please select a valid source type!")