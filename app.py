# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# 페이지이름 설정 
st.set_page_config(
    page_title="Sign Translation using CNNtoLSTM",
    page_icon="🤖"
)

# # Main page heading
# st.title("Sign Translation")
# st.caption('Class 이름 들어가야함')
# st.caption('Then click the :blue[Start] button and check the result.')

# Sidebar
st.sidebar.title("Sign Translation using CNNtoLSTM")
st.sidebar.header("TTS Model Config")

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
# Split the screen into 7:3 ratio
col1, col2 = st.columns([7, 3])


source_img = None
# If image is selected
model = [tts_model, mt_model]

# if source_radio == settings.VIDEO:
#     helper.play_stored_video(time_step, model)

if source_radio == settings.WEBCAM:
    helper.play_webcam(time_step, model, col1, col2)
elif source_radio == settings.VIDEO_FILE:
    helper.play_stored_video(time_step, model, col1, col2)
else:
    st.error("Please select a valid source type!")