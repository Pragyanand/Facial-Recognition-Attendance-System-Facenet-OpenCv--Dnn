import joblib
import cv2, sys
from sklearn.preprocessing import Normalizer
from tensorflow.python.keras.models import load_model
import numpy as np
from train_model import get_embedding
import pickle, time
from mark_attendance import mark_attendance
from create_dataset import face_extractor
import tensorflow as tf
from random import randrange


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


def getTextScale(startX, endX, scale):
    width = endX - startX
    height = 35
    box_area = height * width
    scale = (box_area * scale)/10000
    return scale





def test_model():
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    vc.set(cv2.CAP_PROP_FPS, 30)
    face_model = joblib.load("face_me_chhotu.sav")
    model = load_model('facenet_model/facenet_keras.h5')

    # Opening Label File
    file = open("label_dictionary", 'rb')
    labels = pickle.load(file)
    print(labels)
    global label
    frame_count = 0





    while vc.isOpened():

        ret, frame = vc.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Reading Current Image..", frame)

        if frame_count % 31 != 0:
           frame_count += 1
           continue


        try:



            face, box = face_extractor(frame)

            # face = cv2.resize(face, (160, 160))
            face_array = np.asarray(face)
            face = get_embedding(model, face_array)
            face = face.reshape(1, -1)

            # normalize input vectors
            in_encoder = Normalizer(norm='l2')
            face = in_encoder.transform(face)


            probability = face_model.predict_proba(face)
            argmax = np.argmax(probability[0])
            label_confidence = probability[0, argmax]


            label = labels[argmax]


            id = label.split('_')[-1]
            label = label.split('_')[0]


            rect_color = 0
            if label_confidence >= 0.60 and label_confidence <= 0.85:
                label = "Processing"
                rect_color = (randrange(256), randrange(256), randrange(256))
            else:
                rect_color = (77, 37, 17)

            if label_confidence < 0.60:
                label = "Unknown"
                rect_color = (0, 0, 255)


            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), rect_color, 2)
            cv2.rectangle(frame, (startX, endY-35), (endX, endY), rect_color, -1)

            scale = 1


            cv2.putText(frame, label, (startX+5, endY-8), cv2.FONT_HERSHEY_SIMPLEX, getTextScale(startX, endX, 1.5), (255, 255, 255), 1)
            cv2.imshow("Reading Current Image..", frame)



            if(cv2.waitKey(1) & 0xFF == ord('m')):
                 try:
                     mark_attendance(label, id)
                 except Exception as e:
                    print("Error Marking The Attendance! \n", e)




        except Exception as e:
            cv2.imshow("Reading Current Image..", frame)
            print(sys.exc_info())


        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    vc.release()
    cv2.destroyAllWindows()
    file.close()



if __name__ == '__main__':
    test_model()