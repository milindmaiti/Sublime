import numpy as np
import pickle
from siamese import *
from landmark import *

def predict_gesture(landmarks, gests_to_compare, model, threshold=0.6):
    coords = [[landmark.x, landmark.y] for landmark in landmarks.hand_landmarks[0]]
    coords = np.array(coords)
    assert coords.shape[0] == 21 and coords.shape[1] == 2
    coords = coords.reshape(1, 21, 2)

    highest_sim = 0.0
    highest_id = 0
    for i in range(0, len(gests_to_compare), 2):
        gest_coord1 = gests_to_compare[i].reshape(1, 21, 2)
        gest_coord2 = gests_to_compare[i + 1].reshape(1, 21, 2)
        prediction1 = model.predict([coords, gest_coord1], verbose=False)
        prediction2 = model.predict([coords, gest_coord2], verbose=False)
        avg_prediction = (prediction1 + prediction2) / 2
        # print(avg_prediction)
        if avg_prediction > highest_sim:
            highest_sim = avg_prediction
            highest_id = i / 2
    if highest_sim < threshold:
        return -1 # dissimilar to all gestures
    return int(highest_id)

siamese_net = create_siamese_network((21, 2))
siamese_net.load_weights("siamese_net_v3.h5")

gests_to_compare = pickle.load(open("ground_truths.dat", "rb"))
used_labels = ["dislike", "like", "two_up", "stop", "three2"]  

def add_gesture(coords, new_label, gests_lst, labels_lst): 
    assert coords.shape[0] == 21 and coords.shape[1] == 2
    gests_lst.append(coords)
    print(len(labels_lst))
    if not new_label in labels_lst:
        labels_lst.append(new_label)