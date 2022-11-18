import cv2
import time
import numpy as np
import mediapipe as mp
from collections import Counter
from keras.models import load_model
# from record_speech import record_speech
from keras_preprocessing.image import img_to_array

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'models/model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
def FPS(img, fps, latency):
    cv2.putText(img, f"FPS: {str(int(fps))}", org=(7, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1, color=(0, 0, 0), thickness=1)

    cv2.putText(img, f"Latency: {str(latency)}s", org=(97, 25), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1, color=(0, 0, 0), thickness=1)

    return img

def gen_frames_2():
    global mylist
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                print('face_landmarks:', face_landmarks)
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
            cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

    mylist = []

    pTime, pTimeL = 0, 0
    previous = time.time()
    delta = 0
    message = ""
    a = 0

    cap = cv2.VideoCapture(0)
    cap.set(10, 150)
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while True:
            # _, img = cap.read()
            success, image = cap.read()

            # img = cv2.flip(img, 1)

            image = cv2.resize(image, (640, 480))



            # # FPS
            cTimeL = time.time()

            cTime = time.time()
            if (cTime - pTime) != 0:
                fps = 1 / (cTime - pTime)
                latency = np.round((cTimeL - pTimeL), 4)
                pTime, pTimeL = cTime, cTimeL
                a += 1

                image = FPS(image, fps, latency)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )

                        _, frame = cap.read()
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_classifier.detectMultiScale(gray)

                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                            roi_gray = gray[y:y + h, x:x + w]
                            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                            if np.sum([roi_gray]) != 0:
                                roi = roi_gray.astype('float') / 255.0
                                roi = img_to_array(roi)
                                roi = np.expand_dims(roi, axis=0)

                                prediction = classifier.predict(roi)[0]
                                label = emotion_labels[prediction.argmax()]
                                label_position = (x, y)
                                print(label)
                                mylist.append(label)
                                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Video stream
            ret, buffer = cv2.imencode('.jpg', image)
            image = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


def get_max():
    return max(k for k, v in Counter(mylist).items() if v > 1)


def gen_frames():
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                print('face_landmarks:', face_landmarks)
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
            cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

    mylist = []
    # For webcam input:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)
    capture_duration = 10

    # cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    start_time = time.time()
    # record_speech()
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                    _, frame = cap.read()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_classifier.detectMultiScale(gray)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        roi_gray = gray[y:y + h, x:x + w]
                        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                        if np.sum([roi_gray]) != 0:
                            roi = roi_gray.astype('float') / 255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi, axis=0)

                            prediction = classifier.predict(roi)[0]
                            label = emotion_labels[prediction.argmax()]
                            label_position = (x, y)
                            print(label)
                            mylist.append(label)
                            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
# gen_frames()
# print(max(k for k, v in Counter(mylist).items() if v > 1))
