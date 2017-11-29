import os
import dlib
import glob
from skimage import io
import numpy as np
import json
import pickle

def export_vector_face():
    faces_folder_path = './recognized-faces'
    predictor_path = './shape_predictor_5_face_landmarks.dat'
    face_rec_model_path = './dlib_face_recognition_resnet_model_v1.dat'

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    vector_face = {}
    for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):

        label_int = f.split('/')[-1].split('.')[0]
        img = io.imread(f)
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = sp(img, d)
            vector = np.array(facerec.compute_face_descriptor(img, shape))
            vector_face[label_int] = vector.tolist()

    with open('./recognized-faces/vector_faces.json', 'w') as jsonfile:
        json.dump(vector_face, jsonfile)

def import_vector_face():
    jsonfile = open("./recognized-faces/vector_faces.json")
    
    vector_faces = json.load(jsonfile)

    return vector_faces


export_vector_face()