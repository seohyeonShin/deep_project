from pathlib import Path
import sys
import os

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

SOURCES_LIST = [WEBCAM]

# Images config
# IMAGES_DIR = ROOT / 'images'
# DEFAULT_IMAGE = IMAGES_DIR / 'office_4.jpg'
# DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH
}

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
