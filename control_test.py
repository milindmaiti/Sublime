from time import perf_counter
from inference import *
from landmark import *

# add_gesture("testC1.jpg", "C", gests_to_compare, used_labels)
# add_gesture("testC2.jpg", "C", gests_to_compare, used_labels)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
         # if frame is read correctly ret is True
        if not ret:
             print("Can't receive frame (stream end?). Exiting ...")
             break
         # Our operations on the frame come here
         # Display the resulting frame-by-framecv

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        landmarker.detect_async(mp_image, int(perf_counter() * 1000))

        with lock:
            cur_res = res["result"]

        if cur_res != None:
            hand_landmarks_list = cur_res.hand_landmarks
            # print(hand_landmarks_list)
            if len(hand_landmarks_list) > 0:
                prediction = predict_gesture(cur_res, gests_to_compare, siamese_net)
                if prediction != -1:
                   print(used_labels[prediction])
                print("\n\n")
                for hand_landmark in hand_landmarks_list:
                    pointer_x = hand_landmark[8].x 
                    pointer_y = hand_landmark[8].y
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    pyautogui.moveTo(int(width * pointer_x), int(height * pointer_y))
                    hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmark
                    ])

                    mp_drawing.draw_landmarks(frame_rgb, hand_landmarks_proto, solutions.hands.HAND_CONNECTIONS, solutions.drawing_styles.get_default_hand_landmarks_style(), solutions.drawing_styles.get_default_hand_connections_style())

                    frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
                    

        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
 
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()