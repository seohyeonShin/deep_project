import streamlit as st
import cv2
import torch
from pytube import YouTube
import tempfile
import settings

from weights.tts_weights.inference import PerturbationGradTTS as TTSModel
from weights.mt_weights.inference import MTModel

def load_model(model_path, type='tts'):

    if type == 'tts':
        model = TTSModel(model_path, settings.HIFI_CONF, settings.HIFI_MODEL)
    else:
        # mt model load
        model = MTModel(model_path)
    return model

def play_webcam(time_step, model,col1,col2):

    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    tts_model, mt_model = model
    source_webcam = settings.WEBCAM_PATH
    FRAME_WINDOW = col1.image([])
    file_path = f'recorded_sample.mp4'
    
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    start = st.sidebar.button('Start')
    stop = st.sidebar.button('Stop')
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
            print("Error loading video: " + str(e))
    if stop:
        saved_path_3= "KETI_SL_0000029747.mp4"
        saved_path = "KETI_SL_0000028490.mp4"
        saved_path_2 = "Chest.mp4"
        path = file_path
        text = mt_model.infer(path)
        col2.info(text)

        audio = tts_model.generate_speech(text, time_step)
        col2.audio(audio, format="audio/mp3", sample_rate=22050)


def play_stored_video(time_step, model,col1,col2):
        tts_model, mt_model = model
        source_vid = st.sidebar.selectbox(
            "Choose a video...", settings.VIDEOS_DICT.keys())

        with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            col1.video(video_bytes)
        if st.sidebar.button('Detect Video Objects'):
            try:
                path = str(settings.VIDEOS_DICT.get(source_vid))
                text = mt_model.infer(path)
                col2.info(text)
                # text = '위의 모델을 통해 추론된 테스트 텍스트입니다.'.encode('utf-8')
                audio = tts_model.generate_speech(text, time_step)
                col2.audio(audio, format="audio/mp3", sample_rate=22050)
            except Exception as e:
                print(source_vid)
                print(settings.VIDEOS_DICT.get(source_vid))
                print(e)
def play_upload_video(time_step, model, col1, col2):
    tts_model, mt_model = model
    
    # 비디오 파일 업로드
    uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4"])
    
    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        col1.video(video_bytes)
    
        if st.sidebar.button('Translate Video!'):
            try:
                
                # 저장된 비디오 파일 경로를 사용하여 번역 및 음성 합성 수행
                video_path = uploaded_video
                st.sidebar.info(f"Uploaded video path: {video_path}")
                text = mt_model.infer(uploaded_video.name)
                col2.info(text)
                audio = tts_model.generate_speech(text, time_step)
                col2.audio(audio, format="audio/mp3", sample_rate=22050)
                
            except Exception as e:
                print("Error processing video: " + str(e))