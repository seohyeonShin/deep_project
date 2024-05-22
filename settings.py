from pathlib import Path
import sys
import os
import glob
import json 
import pandas as pd 

# video_list_labe = {"KETI_SL_0000028186.mp4": {"category_id": 0, "name": "가슴"}, "KETI_SL_0000028228.mp4": {"category_id": 1, "name": "귀"}, "KETI_SL_0000028233.mp4": {"category_id": 2, "name": "기절"}, "KETI_SL_0000028238.mp4": {"category_id": 3, "name": "남편"}, "KETI_SL_0000028251.mp4": {"category_id": 4, "name": "누나"}, "KETI_SL_0000028256.mp4": {"category_id": 5, "name": "다리"}, "KETI_SL_0000028268.mp4": {"category_id": 6, "name": "동생"}, "KETI_SL_0000028271.mp4": {"category_id": 7, "name": "두드러기생기다"}, "KETI_SL_0000028284.mp4": {"category_id": 8, "name": "머리"}, "KETI_SL_0000028290.mp4": {"category_id": 9, "name": "무릎"}, "KETI_SL_0000028297.mp4": {"category_id": 10, "name": "발"}, "KETI_SL_0000028314.mp4": {"category_id": 11, "name": "복통"}, "KETI_SL_0000028342.mp4": {"category_id": 12, "name": "손"}, "KETI_SL_0000028349.mp4": {"category_id": 13, "name": "숨을안쉬다"}, "KETI_SL_0000028352.mp4": {"category_id": 14, "name": "심장마비"}, "KETI_SL_0000028353.mp4": {"category_id": 15, "name": "아기"}, "KETI_SL_0000028356.mp4": {"category_id": 16, "name": "아내"}, "KETI_SL_0000028358.mp4": {"category_id": 17, "name": "아빠"}, "KETI_SL_0000028370.mp4": {"category_id": 18, "name": "어깨"}, "KETI_SL_0000028372.mp4": {"category_id": 19, "name": "어지러움"}, "KETI_SL_0000028373.mp4": {"category_id": 20, "name": "언니"}, "KETI_SL_0000028375.mp4": {"category_id": 21, "name": "엄마"}, "KETI_SL_0000028381.mp4": {"category_id": 22, "name": "열"}, "KETI_SL_0000028395.mp4": {"category_id": 23, "name": "오빠"}, "KETI_SL_0000028457.mp4": {"category_id": 24, "name": "피나다"}, "KETI_SL_0000028458.mp4": {"category_id": 25, "name": "친구"}, "KETI_SL_0000028470.mp4": {"category_id": 26, "name": "팔"}, "KETI_SL_0000028478.mp4": {"category_id": 27, "name": "할머니"}, "KETI_SL_0000028479.mp4": {"category_id": 28, "name": "할아버지"}, "KETI_SL_0000028489.mp4": {"category_id": 29, "name": "형"}, "KETI_SL_0000028490.mp4": {"category_id": 30, "name": "호흡곤란"}, "KETI_SL_0000029443.mp4": {"category_id": 0, "name": "가슴"}, "KETI_SL_0000029485.mp4": {"category_id": 1, "name": "귀"}, "KETI_SL_0000029490.mp4": {"category_id": 2, "name": "기절"}, "KETI_SL_0000029495.mp4": {"category_id": 3, "name": "남편"}, "KETI_SL_0000029508.mp4": {"category_id": 4, "name": "누나"}, "KETI_SL_0000029513.mp4": {"category_id": 5, "name": "다리"}, "KETI_SL_0000029525.mp4": {"category_id": 6, "name": "동생"}, "KETI_SL_0000029528.mp4": {"category_id": 7, "name": "두드러기생기다"}, "KETI_SL_0000029541.mp4": {"category_id": 8, "name": "머리"}, "KETI_SL_0000029547.mp4": {"category_id": 9, "name": "무릎"}, "KETI_SL_0000029554.mp4": {"category_id": 10, "name": "발"}, "KETI_SL_0000029571.mp4": {"category_id": 11, "name": "복통"}, "KETI_SL_0000029599.mp4": {"category_id": 12, "name": "손"}, "KETI_SL_0000029606.mp4": {"category_id": 13, "name": "숨을안쉬다"}, "KETI_SL_0000029609.mp4": {"category_id": 14, "name": "심장마비"}, "KETI_SL_0000029610.mp4": {"category_id": 15, "name": "아기"}, "KETI_SL_0000029613.mp4": {"category_id": 16, "name": "아내"}, "KETI_SL_0000029615.mp4": {"category_id": 17, "name": "아빠"}, "KETI_SL_0000029627.mp4": {"category_id": 18, "name": "어깨"}, "KETI_SL_0000029629.mp4": {"category_id": 19, "name": "어지러움"}, "KETI_SL_0000029630.mp4": {"category_id": 20, "name": "언니"}, "KETI_SL_0000029632.mp4": {"category_id": 21, "name": "엄마"}, "KETI_SL_0000029638.mp4": {"category_id": 22, "name": "열"}, "KETI_SL_0000029652.mp4": {"category_id": 23, "name": "오빠"}, "KETI_SL_0000029714.mp4": {"category_id": 24, "name": "피나다"}, "KETI_SL_0000029715.mp4": {"category_id": 25, "name": "친구"}, "KETI_SL_0000029727.mp4": {"category_id": 26, "name": "팔"}, "KETI_SL_0000029735.mp4": {"category_id": 27, "name": "할머니"}, "KETI_SL_0000029736.mp4": {"category_id": 28, "name": "할아버지"}, "KETI_SL_0000029746.mp4": {"category_id": 29, "name": "형"}, "KETI_SL_0000029747.mp4": {"category_id": 30, "name": "호흡곤란"}}
# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
WEBCAM = 'Webcam'
VIDEO_FILE = 'Video File'
UPLOAD = 'Upload'
SOURCES_LIST = [VIDEO_FILE,UPLOAD,WEBCAM]

# Images config
# IMAGES_DIR = ROOT / 'images'
# DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
# DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config


VIDEO_DIR = 'videos'
video_path_list = glob.glob('videos/*.mp4')
with open("./videos/info_video_list.json", encoding='utf-8-sig') as f:
    video_list_Label = json.load(f)
video_name = video_list_Label.keys()
video_pd_Label = pd.DataFrame(video_list_Label)

VIDEOS_ANSWER={}
for video in video_pd_Label.columns:
    VIDEOS_ANSWER[video] = video_pd_Label[video]['name']
VIDEOS_DICT = {}
for video_path in video_path_list:
    video_name = None
    video_file_name = os.path.basename(video_path)
    video_name = VIDEOS_ANSWER[video_file_name]+'_'+video_file_name
    VIDEOS_DICT[video_name] =  video_path



# MT Model config
MT_MODEL_DIR = ROOT / 'weights/mt_weights/checkpts'
MT_MODEL = MT_MODEL_DIR / 'mt_weight.pth'

# TTS Model config
TTS_MODEL_DIR = ROOT / 'weights/tts_weights/checkpts'
TTS_MODEL = TTS_MODEL_DIR / 'grad_444.pt'

# HiFi-GAN Model config
HIFI_MODEL = TTS_MODEL_DIR / 'g_02500000'
HIFI_CONF = TTS_MODEL_DIR / 'config.json'

# Webcam
WEBCAM_PATH = 0
