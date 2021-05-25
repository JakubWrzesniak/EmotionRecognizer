import os
import cv2

emotion_colors={'anger': [255, 0, 0],
                'disgust': [0, 255, 0],
                'fear': [67, 44, 25],
                'happiness': [255, 255, 0],
                'sadness': [0, 0, 0],
                'surprise': [255, 51, 153],
                'neutral': [255, 255, 255]
}

face_cascade = cv2.CascadeClassifier('emotion_recognizer/haarcascade_frontalface_default.xml')

def face_recognize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.05,
                                          minNeighbors=5,
                                          minSize=(48, 48),
                                          flags=cv2.CASCADE_SCALE_IMAGE
                                          )
    fasec_img = []
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        ROI = gray[y:y + h, x:x + w]
        fasec_img.append(ROI)
    return len(faces), fasec_img


def get_color(emotions):
    color = [0, 0, 0]
    for k, v in emotions.items():
        em_color = emotion_colors[k]
        for i in range(3):
            color[i] += em_color[i]*v//100
    return tuple(color)

def emotion_on_img(img, md):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    fasec_img = []
    for i in range(len(faces)):
        (x, y, w, h) = faces[i]
        ROI = gray[y:y + h, x:x + w]
        emotion = process_faces(ROI, md)
        fasec_img.append(ROI)
        color = get_color(emotion)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        emotion = " ".join(":".join([k, str(v) + "%"]) for k, v in emotion.items())
        if emotion:
            cv2.putText(img, emotion, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def process_faces(face, er):
    pr = er.predict(face)
    return pr


def run(model):

    frameWidth = 1280
    frameHeight = 720

    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)

    cap.set(10, 150)
    md = model

    while cap.isOpened():
        try:
            success, img = cap.read()
            img = emotion_on_img(img, md)
            cv2.imshow('img', img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        except:
            pass
    cap.release()
    cv2.destroyAllWindows()


