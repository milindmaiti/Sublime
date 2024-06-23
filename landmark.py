import pyautogui
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from threading import Lock

model_path = "/Users/mmaiti/Documents/programming/screen-control/hand_landmarker.task"
width, height = pyautogui.size()
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.solutions.drawing_utils
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()   

res = {"result": None}
lock = Lock()
# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    with lock:
        res["result"] = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/Users/mmaiti/Documents/programming/screen-control/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)