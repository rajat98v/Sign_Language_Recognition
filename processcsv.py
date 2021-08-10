import mediapipe as mp
import cv2


import pandas as pd
import os
import pandas as pd
import numpy as np
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.utils import to_categorical, normalize

from sklearn.model_selection import train_test_split

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

def trainsave():
    training_data = pd.read_csv("hand_datas.csv")

    features = training_data.copy()
    string_labels = np.array(features.pop('class'))

    # Get only unique labels from list of all label class names
    unique_labels = np.unique(string_labels)

    # Save Labels in file for use in prediction
    file = open("arr", "wb")
    np.save(file, unique_labels)
    file.close()

    integer_mapping = {x: i for i,x in enumerate(unique_labels)}
    vec = []
    for values in string_labels :
        vec.append(integer_mapping[values])

    # Converts a class vector (integers) to binary class matrix # one hot encoding
    labels = to_categorical(vec)

    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.33,
                                                        random_state=42,
                                                        shuffle=True,
                                                        )

    model = Sequential()
    model.add(Dense(42, activation='relu'),)
    model.add(Dense(64 ,activation='relu'))
    model.add(Dense(64 ,activation='relu'))
    model.add(Dense(len(unique_labels), activation='softmax'))

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


    #function is a convenient way of letting the model train until an optimum is found
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=10,
        restore_best_weights=True)

    model.fit(X_train, y_train,
          # batch_size=1000,
          epochs=50,
          verbose=1,
          shuffle=True,
          validation_data=(X_test, y_test),
          callbacks=[es])

    # model.fit(X_train, y_train, epochs=100)
    # model.evaluate(X_test,  y_test, verbose=2)

    model.save("mymodel2")


trainsave()

model = tf.keras.models.load_model("mymodel2")

#read the file to numpy array
file = open("arr", "rb")
class_list = np.load(file, allow_pickle=True)

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('rtsp://25.174.108.223:8080/h264_ulaw.sdp')
# capture = cv2.VideoCapture('rtsp://25.174.108.223:8080/1')

cv2.imshow('image', cv2.imread('asl.jpg'))

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

with mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = hand_landmarks.landmark
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = list(np.array([[landmark.x, landmark.y] for landmark in coords]).flatten())
                # coords = scaler.transform([coords])
                
                # Alternative for dataset using z coordinates.
                # Z coordinates is not recommended, since you need to adjust your distance from camera.
#                 coords = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in coords]).flatten())
                rgb_tensor = tf.convert_to_tensor(coords, dtype=tf.float32)

                ##Add dims to rgb_tensor
                rgb_tensor = tf.expand_dims(rgb_tensor , 0)
                predict = model.predict(rgb_tensor, steps=1)
                predicted = class_list[np.argmax(predict[0])]
                
                # predicted = model.predict(coords)

            # Get status box
            cv2.rectangle(image, (0,0), (100, 60), (245, 90, 16), -1)

            # Display Class
            cv2.putText(image, 'CLASS'
                        , (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(predicted[0])
                        , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Hands', image)

        # Press esc to close webcam
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()

# processing file:  /content/ASL_Dataset/Train/D/D2394.jpg
