import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import mediapipe as mp
import pyaudio
import wave
import threading

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands # 손동작 인식 


class VideoTransformer(VideoTransformerBase):
    # transform 내부에서 프레임필터를 구현할수 있음
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

        return img

def main():
    # st.title("Webcam Video Capture with Audio Playback")
    # st.write("Press the button below to start capturing video from your webcam.")
    # st.audio("audio.wav", format="audio/mpeg", loop=True)
    # st.title("Webcam Live Feed")
    # run = st.checkbox('Run')
    # FRAME_WINDOW = st.image([])
    # camera = cv2.VideoCapture(0)

    # while run:
    #     _, frame = camera.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     FRAME_WINDOW.image(frame)
    # else:
    #     st.write('Stopped')
    # webrtc_streamer(key="example")
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
if __name__ == "__main__":
    main()
