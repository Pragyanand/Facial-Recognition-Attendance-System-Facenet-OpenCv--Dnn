from pprint import pprint

from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, Normalizer
import joblib
import cv2,os
import numpy as np
import pickle





def load_dataset():
    data_path = "dataset/"
    dataset_folder = os.listdir(data_path)

    faces = []
    labels = []
    label_dict = {}
    key = 0

    for user in dataset_folder:
        label_dict[key] = user
        user_path = data_path + user + "/"
        files = os.listdir(user_path)
        key += 1

        for file in files:
            if file.endswith(".jpg"):
                file_path = user_path + file
                image = cv2.imread(file_path)
                image = cv2.resize(image, (160, 160))
                image = np.asarray(image)
                faces.append(image)

                filename = file.split('_')[0]
                labels.append(filename)

    filehandler = open("label_dictionary", 'wb')
    pickle.dump(label_dict, filehandler)
    filehandler.close()

    np.savez_compressed('compressed_dataset.npz', faces, labels)


    return np.asarray(faces), np.asarray(labels)




# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


def training_model():
    try:
        faces, labels = load_dataset()
        print("All the Data is Loaded!")
    except Exception as o:
        print(o)

    data = np.load('compressed_dataset.npz', allow_pickle=True)
    faces, labels = data['arr_0'], data['arr_1']



    trainX, testX, trainy, testy = train_test_split(faces, labels, test_size=0.20, random_state=1)




    # load the facenet model
    model = load_model('facenet_model/facenet_keras.h5')
    print('Facenet Keras Model Loaded (for Embedding Creation!)')


    # convert each face in the train set to an embedding
    newTrainX = list()
    count = 0
    for face_pixels in trainX:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = np.asarray(newTrainX)

    # convert each face in the test set to an embedding
    newTestX = list()
    for face_pixels in testX:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)
    newTestX = np.asarray(newTestX)
    # save arrays to one file in compressed format
    np.savez_compressed('face_embeddings.npz', newTrainX, trainy, newTestX, testy)

    print("Embedding Creation is Done!")


    ## FACE  CLASSIFICATION


    # load dataset
    data = np.load('face_embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)



    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)

    print("Label Encoding is Completed!")



    # fit model
    model = RandomForestClassifier(n_estimators=1000,
                                   n_jobs= -1,
                                   random_state= 44,
                                   max_features = 10)


    print("****Model Training Started****")
    model.fit(trainX, trainy)
    print("****Model Training Completed!****")



    # predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)



    # score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)



    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

    joblib.dump(model, "face_me_chhotu.sav")



if __name__ == '__main__':
    training_model()
