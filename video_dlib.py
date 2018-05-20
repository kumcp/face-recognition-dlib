import numpy as np
import cv2
import dlib
import json

from performance_measure import measure_time
from create_facedata import import_vector_face
import time 

cap = cv2.VideoCapture(0)

predictor_path = './shape_predictor_5_face_landmarks.dat'
face_rec_model_path = './dlib_face_recognition_resnet_model_v1.dat'


# detector = dlib.get_frontal_face_detector()
detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')

sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

skipframe = 10

jsonfile = open("./recognized-faces/label_map.json")
    
label_map = json.loads(jsonfile.read())
threshold = 0.1
likely = 0.2

def detect_realtime():
    i=0
    old_faces = []
    while(cap.isOpened()):

        ret, frame = cap.read()
        
        faces_detected = []

        if i == skipframe:
            i=0
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            start_time = time.time()
            dets = detector(frame, 1)
            end_time = time.time()

            print("Detecting takes: %f ms \n" % ((end_time-start_time) *1000) )

            new_faces = []

            for k, d in enumerate(dets):
                # Remove this line if using get_frontal_face_detector()
                d = d.rect
                #######
                
                shape = sp(frame, d)
                face_descriptor = facerec.compute_face_descriptor(frame, shape)
                name = find_same_face(face_descriptor)
                new_faces.append((d.left(), d.top(), d.right()-d.left(), d.bottom() - d.top(), name))

                print("Detection: {} Name :{} Left: {} Top: {} Right: {} Bottom: {}".format(
                    k, name ,d.left(), d.top(), d.right(), d.bottom()))

            for (x, y, w, h, name) in new_faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            old_faces = new_faces[:]
        else:
            i=i+1
            for (x, y, w, h, name) in old_faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x-2, y-2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        if len(faces_detected) >0:
            print("This is : {}".format(faces_detected))
        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

@measure_time
def find_same_face(face_descriptor):
    recognized_faces = import_vector_face()

    face_vector = np.array(face_descriptor)

    min_loss = 1.0
    most_likely_face = "Unknown"

    for face_name in recognized_faces:
        loss = np.sum((face_vector - np.array(recognized_faces[face_name])) ** 2)
        # print(loss)
        if loss < min_loss:
            min_loss = loss
            most_likely_face = label_map[face_name]

        if loss <threshold:
            return label_map[face_name]
    
    if min_loss < likely:
        return most_likely_face+"?"

    return "Unknown"

detect_realtime()