import cv2
import numpy as np
import os

# loads base image for now to test
image = cv2.imread("image2.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the trained facial recognition model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.FONT_HERSHEY_SIMPLEX
)


if len(faces) > 0:
    (x, y, w, h) = faces[0]
    face_roi = gray_image[y:y+h, x:x+w]
    face_roi = cv2.resize(face_roi, (200, 200))
    face_roi = face_roi / 255.0
    face_roi = face_roi.flatten()
    face_roi = face_roi.reshape(1, -1)
    name = input("Enter a name for this face: ")
    np.save("faces/" + name + ".npy", face_roi)


cap = cv2.VideoCapture(0)

amount = 0

r = 0,
g = 255,
b = 0,

while True:

    ret, frame = cap.read()
    #frame = cv2.imread("image.jpg")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    print("Faces: " + str(len(faces)))
    if len(faces) > 0:
        for i in range(len(faces)):
            (x, y, w, h) = faces[i-1]

            face_roi = gray_frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_roi = face_roi / 255.0
            face_roi = face_roi.flatten()
            face_roi = face_roi.reshape(1, -1)

            saved_faces = {}
            for file in os.listdir("faces"):
                name, _ = file.split(".")
                face = np.load("faces/" + file)
                saved_faces[name] = face

            min_distance = 50
            name = None

            for key, value in saved_faces.items():
                distance = np.linalg.norm(face_roi - value)
                if distance < min_distance:
                    min_distance = distance
                    name = key
                    
            print("Distance: " + str(min_distance))
            if min_distance < 40:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (r, g, b), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()