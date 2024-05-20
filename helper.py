import streamlit as st
import cv2
import torch
from pytube import YouTube

import settings

from weights.tts_weights.inference import PerturbationGradTTS as TTSModel
from weights.mt_weights.inference import MTModel


def load_model(model_path, type='tts'):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    if type == 'tts':
        model = TTSModel(model_path, settings.HIFI_CONF, settings.HIFI_MODEL)
    else:
        # mt model load
        model = MTModel(model_path)
    return model

def play_webcam(time_step, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    tts_model, mt_model = model
    source_webcam = settings.WEBCAM_PATH
    FRAME_WINDOW = st.image([])
    file_path = f'recorded_sample.mp4'
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    start = st.button('Start')
    stop = st.button('Stop')
    frames = []
    if start:
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            width = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            size = (int(width), int(height))
            out = cv2.VideoWriter(file_path, fourcc, fps, size)

            is_run = True
            while (vid_cap.isOpened() and is_run):
                success, image = vid_cap.read()
                if success:
                    # 기존의 경우 Real-Time Detection
                    # 해당 프로젝트에서는 Real-Time Text 추론 및 음성 생성
                    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame)
                    out.write(frame)
                else:
                    vid_cap.release()
                    out.release()
                    break

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
    if stop:
        path = file_path
        text = mt_model.infer(path)
        st.write(text)

        audio = tts_model.generate_speech(text, time_step)
        st.audio(audio, format="audio/mp3", sample_rate=22050)

#
# def play_stored_video(time_step, model):
#     """
#     Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.
#
#     Parameters:
#         conf: Confidence of YOLOv8 model.
#         model: An instance of the `YOLOv8` class containing the YOLOv8 model.
#
#     Returns:
#         None
#
#     Raises:
#         None
#     """
#     tts_model, mt_model = model
#     source_vid = st.sidebar.selectbox(
#         "Choose a video...", settings.VIDEOS_DICT.keys())
#
#
#     with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
#         video_bytes = video_file.read()
#     if video_bytes:
#         st.video(video_bytes)
#
#     if st.sidebar.button('Detect Video Objects'):
#         try:
#             vid_cap = cv2.VideoCapture(
#                 str(settings.VIDEOS_DICT.get(source_vid)))
#             st_frame = st.empty()
#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if success:
#                     # 기존의 경우 Real-Time Detection
#                     # 해당 프로젝트에서는 Real-Time Text 추론 및 음성 생성
#                     pass
#                 else:
#                     vid_cap.release()
#                     break
#             text = '위의 모델을 통해 추론된 테스트 텍스트입니다.'.encode('utf-8')
#             audio = tts_model.generate_speech(text, time_step)
#             st.audio(audio, format="audio/mp3")
#         except Exception as e:
#             st.sidebar.error("Error loading video: " + str(e))
