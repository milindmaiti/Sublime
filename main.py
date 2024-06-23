"""
pygame-menu
https://github.com/ppizarror/pygame-menu

EXAMPLE - GAME SELECTOR
Game with 3 difficulty options.
"""

from time import perf_counter
__all__ = ['main']

import pygame
import pygame_menu
import time
from inference import *

import cv2
from pygame_menu.examples import create_example_window
from mediapipe import solutions

from random import randrange
from typing import Tuple, Any, Optional, List

# Constants and global variables

ABOUT = [f"""How to use: Click Add Gesture 
    to take a picture of a gesture you 
    want to add, and then simply select 
    a command that you would like to map 
    that gesture to in order to add a 
    gesture. Simply click activate to 
    begin controlling your screen with 
    gestures, with your index finger 
    controlling the mouse.""",
         f'Authors: Milind Maiti, Christopher Sun']
DIFFICULTY = ['EASY']


COMMANDS = ['CLICK', 'SCROLL_UP', 'SCROLL_DOWN', 'ZOOM_IN', 'ZOOM_OUT']
GESTURES = ['dislike', 'like', 'two_up', 'stop', 'three2']
SELECTOR_IDS = [i for i in range(len(COMMANDS))]
COMMANDS_LIST = [(COMMANDS[i], str(i)) for i in range(len(COMMANDS))]
FPS = 60


mappings = dict([(GESTURES[i], COMMANDS[i]) for i in range(len(COMMANDS))])
WINDOW_SIZE = (640, 480)
APP_NAME = 'Sublime'

clock: Optional['pygame.time.Clock'] = None
main_menu: Optional['pygame_menu.Menu'] = None
surface: Optional['pygame.Surface'] = None
edit_gesture_menu: Optional['pygame_menu.Menu'] = None
img_cache = []


def change_mapping(value: Tuple[Any, Any], *args, **kwargs) -> None:
    selected, index = value
    mappings[GESTURES[int(kwargs['sid'])]] = COMMANDS[index]

def addGesture(text):
    global edit_gesture_menu
    global img_cache
    mappings[text] = 'CLICK'

    add_gesture(img_cache[0], text, gests_to_compare, GESTURES)
    add_gesture(img_cache[1], text, gests_to_compare, GESTURES)
    
    edit_gesture_menu.add.selector(text, COMMANDS_LIST, default=0, onchange=change_mapping, sid=str(len(GESTURES)-1))
    main_menu._open(edit_gesture_menu)
    
def videoCapture():
    global main_menu
    global edit_gesture_menu
    global clock
    global cap
    global input_menu
    global img_cache

    count = 0
    main_menu.disable()
    cap = cv2.VideoCapture(0)
    add = 800
    surface = create_example_window(APP_NAME, (WINDOW_SIZE[0]+add, WINDOW_SIZE[1]+add))
    img_cache = []
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    while True:

        clock.tick(60)
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pygame = pygame.image.frombuffer(frame_rgb.tobytes(), frame.shape[1::-1], 'RGB')
        pygame.display.flip()
        # surface.fill((255,255,255))
        surface.blit(frame_pygame, (0, 0))
        
        # If frame is read correctly, ret is True
        
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                exit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    with HandLandmarker.create_from_options(options) as landmarker:
                        landmarker.detect_async(mp_image, int(perf_counter() * 1000))

                        with lock:
                            cur_res = res["result"]

                        if(cur_res != None):
                            cur_res = np.array([[landmark.x, landmark.y] for landmark in cur_res.hand_landmarks[0]])
                            img_cache.append(cur_res)
                            
                            surface.fill((255,255,255))
                            pygame.display.flip()
                            time.sleep(0.1)
                            count += 1
                            print("The count is", count)
                    if(count == 2):
                        print(img_cache)
                        main_menu.enable()
                        main_menu._open(input_menu)
                        
                        cap.release()
                        # Close all OpenCV windows
                        cv2.destroyAllWindows()
                        surface = create_example_window(APP_NAME, WINDOW_SIZE)
                        return
                

def appLoop():
    global main_menu
    global edit_gesture_menu
    global clock
    global cap

    main_menu.disable()
    cap = cv2.VideoCapture(0)
    add = 800
    surface = create_example_window(APP_NAME, (WINDOW_SIZE[0]+add, WINDOW_SIZE[1]+add))
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    while True:

        clock.tick(60)
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        with HandLandmarker.create_from_options(options) as landmarker:
            landmarker.detect_async(mp_image, int(perf_counter() * 1000))

            with lock:
                cur_res = res["result"]

            if cur_res != None:
                hand_landmarks_list = cur_res.hand_landmarks
                # print(hand_landmarks_list)
                if len(hand_landmarks_list) > 0:
                    prediction = predict_gesture(cur_res, gests_to_compare, siamese_net)
                    print("\n\n")
                    for hand_landmark in hand_landmarks_list:
                        pointer_x = hand_landmark[8].x 
                        pointer_y = hand_landmark[8].y
                        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

                        if(prediction == -1):
                            pyautogui.moveTo(int(width * pointer_x), int(height * pointer_y))
                        else:
                            action = mappings[GESTURES[prediction]]

                            if(action == 'CLICK'):
                                pyautogui.click()
                            elif(action == 'SCROLL_UP'):
                                pyautogui.scroll(10)
                            elif(action == 'SCROLL_DOWN'):
                                pyautogui.scroll(-10)
                            elif(action == 'ZOOM_IN'):
                                pyautogui.hotkey('command', '+')
                            elif(action == 'ZOOM_OUT'):
                                pyautogui.hotkey('command', '-')
                        hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmark
                        ])

                        mp_drawing.draw_landmarks(frame_rgb, hand_landmarks_proto, solutions.hands.HAND_CONNECTIONS, solutions.drawing_styles.get_default_hand_landmarks_style(), solutions.drawing_styles.get_default_hand_connections_style())

                        frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        frame_pygame = pygame.image.frombuffer(frame_rgb.tobytes(), frame.shape[1::-1], 'RGB')
        pygame.display.flip()
        surface.fill((0,0,0))
        surface.blit(frame_pygame, (0, 0))


        
        # If frame is read correctly, ret is True

        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                exit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    main_menu.enable()
                    
                    cap.release()
                    # Close all OpenCV windows
                    cv2.destroyAllWindows()
                    surface = create_example_window(APP_NAME, WINDOW_SIZE)
                    return
                

def main_background() -> None:
    """
    Function used by menus, draw on background while menu is active.
    """
    global surface
    surface.fill((128, 0, 128))


def main(test: bool = False) -> None:
    """
    Main program.

    :param test: Indicate function is being tested
    """

    
    # -------------------------------------------------------------------------
    # Globals
    # -------------------------------------------------------------------------
    global clock
    global main_menu
    global edit_gesture_menu
    global surface
    global input_menu

    surface = create_example_window(APP_NAME, WINDOW_SIZE)
    clock = pygame.time.Clock()

    submenu_theme = pygame_menu.themes.THEME_DEFAULT.copy()
    submenu_theme.widget_font_size = 15
    edit_gesture_menu = pygame_menu.Menu(
        height=WINDOW_SIZE[1] * 0.5,
        theme=submenu_theme,
        title='Submenu',
        width=WINDOW_SIZE[0] * 0.7
    )
    for i in range(len(COMMANDS)):
        edit_gesture_menu.add.selector(GESTURES[i], COMMANDS_LIST, default=i, onchange=change_mapping, sid=str(i))

    # -------------------------------------------------------------------------
    # Create menus:About
    # -------------------------------------------------------------------------
    about_theme = pygame_menu.themes.THEME_DEFAULT.copy()
    about_theme.widget_margin = (0, 0)

    about_menu = pygame_menu.Menu(
        height=WINDOW_SIZE[1] * 0.6,
        theme=about_theme,
        title='About',
        width=WINDOW_SIZE[0] * 0.6
    )

    input_menu = pygame_menu.Menu(
        height=WINDOW_SIZE[1] * 0.6,
        theme=about_theme,
        title='Gesture Title',
        width=WINDOW_SIZE[0] * 0.6
    )

    input_menu.add.text_input("Gesture Label: ", onreturn = addGesture)

    for m in ABOUT:
        about_menu.add.label(m, align=pygame_menu.locals.ALIGN_LEFT, font_size=20)
    about_menu.add.vertical_margin(30)
    about_menu.add.button('Return to menu', pygame_menu.events.BACK)

    # -------------------------------------------------------------------------
    # Create menus: Main
    # -------------------------------------------------------------------------
    main_theme = pygame_menu.themes.THEME_DEFAULT.copy()

    main_menu = pygame_menu.Menu(
        height=WINDOW_SIZE[1] * 0.6,
        theme=main_theme,
        title='Main Menu',
        width=WINDOW_SIZE[0] * 0.6
    )

    gesture_theme = pygame_menu.themes.THEME_BLUE.copy()

    main_menu.add.button('Activate App', appLoop)
    main_menu.add.button('Add Gesture', videoCapture)
    main_menu.add.button('About', about_menu)
    main_menu.add.button('Gesture Settings', edit_gesture_menu)
    main_menu.add.button('Exit', pygame_menu.events.EXIT)

    

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    while True:

        # Tick
        clock.tick(FPS)

        # Paint background
        main_background()

        # Application events
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                exit()

        # Main menu
        if main_menu.is_enabled():
            main_menu.mainloop(surface, main_background, disable_loop=test, fps_limit=FPS)

        # Flip surface
        pygame.display.flip()

        # At first loop returns
        if test:
            break


if __name__ == '__main__':
    main()

