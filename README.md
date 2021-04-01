
# Summary


The project aims at creating an Attendance System that marks the attendance of people by using a live webcam feed and store that into a mysql database.


## Tools Used :

* IDE :  Pycharm
* Language : Python 3.7.9



* Note : Remember to use Virtualenv whenever creating a project.<br><br>
    * Helps in creating a good requirements.txt file. ( pip freeze > requirements.txt)
    * Use WindDirStat for visualizing project directory.


## Project Tree



![folder_tree-3.png](https://github.com/Pragyanand/Facial-Recognition-Attendance-System/blob/main/folder_tree.png)



*  .idea, pycache, venv : These folders are created by pycharm IDE.
* dataset : contains the dataset, when it is created.





* face_detection_model
    * deploy.prototxt : defines the model architecture i.e. the layers of the dnn.
    * dnn_model.caffemodel : contains the weights for the layers. <br><br>
    * This is a deep learning based face_detection model that comes with the open_cv module, you just need to download these two files.
<br><br>

- Before finalizing this model for detecting faces, i used
    - HaarCascascade Model
    - MTCNN <br><br>
    
    * HaarCascade Model was discarded because it coudn't detect side faces or partial faces.
    * MTCNN was discarded because it was too slow for the purpose.


* face_model
    * facenet_keras.h5 : Pre_trained Model; Provides numerical embeddings on the imgages fed to it.
    * I haven't included here but can be found in just one simple google search.
    
        FaceNet is a system that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. Once this space has been produced, tasks such as face recognition, verification and clustering can be easily implemented using standard techniques with FaceNet embeddings as feature vectors.
    
    FaceNet maps a face into a 128D Euclidien space. The L2 distance(or Euclidien norm) between two faces embeddings corresponds to its similarity. This is exactly like measuring the distance between two points in a line to know if they are close to each other.*
    
Here's the link to the [FaceNet Paper](https://arxiv.org/abs/1503.03832).



* Files :
    * create_dataset.py :
        - Opens the Webcam
        - Asks for Name and Id:
        - Proceeds to Capture 100 images while you turn your head in different directions.
        - saves them in a folder named 'username_id' inside dataset folder. <br><br>
        
    * data_augmentation.py
        - performs 5 levels of Brightness and Contrast adjustments on all 100 images.
        - Also performs horizontal flip and 4 level of rotations in step_sizes = 90 degress for all the images         generated after brightness and contrast adjustments. <br><br>
        
    * train_model.py
        - trains the model on the all the data generated.
        - Goes into each folder of the dataset and gets the name and id of the person from the folder name.
        - Uses the names as the label and user labelencoder to convert them into numbers and creates a list according the number of picutres in that folder.
        - All the images are fed to the facenet model to get their embeddings and a list is created.
        - After all folders are done, both labels and image_embeddings are them compressed and saved as '.npz' file.
        - I have used Knn classifier with default parameters to classify the faces. (first google search suggested me to use knn, so i did, also it provies the probability of the correct classification, which was very useful when testing the model.)
        - knn model is saved after the training is finised.
        <br><br>
        
    * test_model.py:
        - Opens up the webcam.
        - Loads the **face_detection model**, frames are fed to it, returns cropped faces.
        - Faces are then fed to **facenet model** and embeddings are returned.
        - knn model is loaded and these embeddings are then fed to it predictions are returned and displayed on the frame.<br><br>
    * connect.py : 
        - defines the connection to the mysql database. <br><br>

    * mark_attendance : 
        - defines a function which saves the name, id, time and date of attendance into a table of a **MySql** database.<br><br>
        
    * label_dictionary :
        - This is created at the time of dataset creation to store the names against the label encoder's encoding to later display the person's name instead of number when our model is being tested. 


## Drawbacks


* It can be fooled using a photograph or a video.
    - Blink test or randomly specified head movement test can be implemented to prevent this from happening. <br><br>
* Change in lighting conditions and distance and posture of the face in accordance with the camera can affect the model's prediction performance. <br><br>
* Model Training takes a lot of time.<br><br>



