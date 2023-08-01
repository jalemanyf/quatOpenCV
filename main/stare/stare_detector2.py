import cv2
import dlib
import numpy as np 
import math
from retinaface import RetinaFace
from deepface import DeepFace

class StareDetector:
    def __init__(self):
        # self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        self.unique_embeddings = []
        self.threshold_angle = 20

    def reset():
        self.unique_embeddings = []

    def check_frame(self, frame):
        '''
        Count the people that is staring into the camera in the frame.
        Saves the face embeddings so same people don't get counted twice.
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = self.detector(gray)
        # faces_data = RetinaFace.detect_faces(frame)
        retinaface_detected_faces = RetinaFace.detect_faces(frame)
        #print("retinaface_detected_faces:", retinaface_detected_faces)
        faces = []
        if isinstance(retinaface_detected_faces, dict):
            offset = 25
            faces = [
                dlib.rectangle(
                    max(df['facial_area'][0]-offset, 0),
                    max(df['facial_area'][1]-offset, 0),
                    min(df['facial_area'][2]+offset, frame.shape[0]),
                    min(df['facial_area'][3]+offset, frame.shape[1])
                    # df['facial_area'][0],
                    # df['facial_area'][1],
                    # df['facial_area'][2],
                    # df['facial_area'][3],
                )
                for df in retinaface_detected_faces.values()
            ]
        faces_info = []
        for face in faces:
            landmarks = self.predictor(gray, face)
            left_eye_left_corner = (landmarks.part(36).x, landmarks.part(36).y)
            left_eye_right_corner = (landmarks.part(39).x, landmarks.part(39).y)
            right_eye_left_corner = (landmarks.part(42).x, landmarks.part(42).y)
            right_eye_right_corner = (landmarks.part(45).x, landmarks.part(45).y)
            dx = right_eye_right_corner[0] - left_eye_left_corner[0]
            dy = right_eye_right_corner[1] - left_eye_left_corner[1]
            center = [
                (right_eye_right_corner[0] + left_eye_left_corner[0]) //2,
                (right_eye_right_corner[1] + left_eye_left_corner[1]) //2
            ]
            angle = math.degrees(math.atan2(dy, dx))
            if abs(angle) < self.threshold_angle:
                look_status = "Looking at Screen"
            else:
                look_status = "Not Looking at Screen"
            face_embedding = self.facerec.compute_face_descriptor(frame, landmarks)
            is_unique = True
            for emb_idx, emb in enumerate(self.unique_embeddings):
                distance = np.linalg.norm(np.array(face_embedding) - np.array(emb))
                if distance < 0.7:
                    is_unique = False
                    face_id = emb_idx
                    break
            if is_unique:
                face_id = len(self.unique_embeddings)
                self.unique_embeddings.append(face_embedding)
            
            colour_face = frame[face.top():face.bottom(), face.left():face.right()]
            #print(type(colour_face))
            obj= DeepFace.analyze(colour_face,detector_backend='skip',enforce_detection=False)
            #print("obj:", obj)
            if len(obj) > 0: # Check if obj is not empty
                dominant_emo = obj[0]['dominant_emotion']
                emotions = obj[0]['emotion']
                dominant_race = obj[0]['dominant_race']
                age = int(obj[0]['age'])
                gender = obj[0]['dominant_gender']
                gender_prob = obj[0]['gender']
      
                faces_info.append({
                    'face_id': face_id,
                    'center': center,
                    'dominant_emotion': dominant_emo,
                    'emotions': emotions,
                    'dominant_race': dominant_race,
                    'age': age,
                    'gender': gender,
                    'gender_prob':gender_prob,
                    'pos':{
                        'top': face.top(),
                        'bottom': face.bottom(),
                        'left': face.left(),
                        'right': face.right()
                    }
                })
        return {
            'unique_embeddings': len(self.unique_embeddings),
            'faces_info': faces_info,
        }

