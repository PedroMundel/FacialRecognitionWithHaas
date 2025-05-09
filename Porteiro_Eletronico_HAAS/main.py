import cv2, os, json

LABELS_FILE = "labels.json"

def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return json.load(f)
    return {}

labels = load_labels()


def draw_boundary(img, classifier, scaleFactor, minNeighbours, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)
        id, confidence = clf.predict(gray_img[y:y+h, x:x+w])
        name = labels.get(str(id), f"Unknown_{id}")
        cv2.putText(img, name, (x, y-4), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


def recognize(img, clf, face_cascade):
   color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)} 

   coords = draw_boundary(img, face_cascade, 1.1, 7, color["red"], "Face", clf)
   return img

def detect(img, face_cascade, eye_cascade, nose_cascade, mouth_cascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, face_cascade, 1.1, 5, color["red"], "Face")

    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        coords = draw_boundary(roi_img, eye_cascade, 1.1, 7, color["blue"], "Eyes")
        coords = draw_boundary(roi_img, nose_cascade, 1.1, 9, color["green"], "Nose")
        coords = draw_boundary(roi_img, mouth_cascade, 1.1, 25, color["white"], "Mouth")
    return img


nose_path = os.path.abspath("haarcascade_mcs_nose.xml")
mouth_path = os.path.abspath("haarcascade_mcs_mouth.xml")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier(nose_path)
print("Loaded:", nose_cascade.empty())
mouth_cascade = cv2.CascadeClassifier(mouth_path)
print("Loaded:", mouth_cascade.empty())


clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

video_capture = cv2.VideoCapture(0)

img_id = 0

while True:
    _, img = video_capture.read()
    #img = detect(img, face_cascade, eye_cascade, nose_cascade, mouth_cascade)
    img = recognize(img, clf, face_cascade)
    if not _:
        break

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()