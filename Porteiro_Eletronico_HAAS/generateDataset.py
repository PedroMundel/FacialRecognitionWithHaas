import cv2
import os
import json

LABELS_FILE = "labels.json"
DATA_DIR = "data"

def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_labels(labels):
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f)

def get_next_id(labels):
    if not labels:
        return 1
    return max(int(uid) for uid in labels.keys()) + 1

def generate_dataset(img, user_id, img_id):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    cv2.imwrite(f"{DATA_DIR}/User.{user_id}.{img_id}.jpg", img)

def draw_boundary(img, classifier, scaleFactor, minNeighbours, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbours)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def detect(img, face_cascade, img_id, user_id):
    color = {"red": (0, 0, 255)}
    coords = draw_boundary(img, face_cascade, 1.1, 5, color["red"], "Face")
    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        generate_dataset(roi_img, user_id, img_id)
    return img

# Load or create name-ID mapping
labels = load_labels()

# Assign a new user ID
user_id = get_next_id(labels)
name = input("Enter your name: ")
labels[str(user_id)] = name
save_labels(labels)

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
img_id = 0

print(f"[INFO] Capturing images for '{name}' (User ID {user_id}). Press 'q' to stop.")

while True:
    ret, img = video_capture.read()
    if not ret:
        break
    img = detect(img, face_cascade, img_id, user_id)
    cv2.imshow("Face Capture", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
