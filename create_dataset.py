import errno
import cv2, os, time, pickle
import numpy as np

net = cv2.dnn.readNetFromCaffe("face_detection_model/deploy.prototxt", "face_detection_model/dnn_model.caffemodel")



def face_extractor(frame):



    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    b_mean = np.mean(frame[:, :, 0])
    g_mean = np.mean(frame[:, :, 1])
    r_mean = np.mean(frame[:, :, 2])

     # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (r_mean, g_mean, b_mean), swapRB=True)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # Iterate over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.80:
            continue

        # Compute coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, (160, 160))
        return face, box





def create_dataset():


    count = 1

    username = input("Enter your name : ")

    while True:
        try:
            id = int(input("Enter your id :"))
            if isinstance(id, int) == True:
                break

        except:
            print("Id should only be string!")
            print("Try Again and Only Input Integer Values!")

    data_path = "/dataset/"

    start_key = False


    try:


        vc = cv2.VideoCapture(0)

        project_directory = os.path.abspath(os.curdir)
        user_folder = project_directory + data_path + username + "_" + str(id)




        while vc.isOpened():

            filename = username + "_" + str(id) + "_" + str(count)+ ".jpg"

            ret, frame = vc.read()
            frame = cv2.flip(frame, 1)

            try:

                if not os.path.exists(user_folder):
                    print("Directory Doesn't Exists, Creating Directory!")
                    os.makedirs(user_folder)
                    os.chdir(user_folder)
                    print(os.getcwd())
                else:
                    os.chdir(user_folder)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            cv2.imshow("Frame", frame)

            if start_key == False:
                key = cv2.waitKey(1)

            if (key & 0xFF == ord('s')):

                try:
                    start_key = True
                    face, box = face_extractor(frame)
                    time.sleep(0.5)
                    cv2.imwrite(filename, face)
                    count +=1


                    (startX, startY, endX, endY) = box.astype("int")

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 255), 3)
                    cv2.putText(frame, str(count), (endX // 4, endY // 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 255, 0), 2)
                    cv2.imshow("Frame", frame)

                    print("capturing images")


                except:
                    cv2.putText(frame, "No Face in the Frame", (endX//4, endY//20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Frame", frame)

            if (cv2.waitKey(1) & 0xFF == ord('q') or count > 100):
                break

        vc.release()
        cv2.destroyAllWindows()
        # file.close()


    except Exception as e:
        print(e)


if __name__ == '__main__':
    create_dataset()