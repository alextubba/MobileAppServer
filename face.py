import cv2
import numpy as np
import os

# Load the static image and convert it to grayscale
image = cv2.imread("image.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the trained facial recognition model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(
    gray_image,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.FONT_HERSHEY_SIMPLEX
)

# Check if a face was detected
if len(faces) > 0:
    # Extract the coordinates of the first face
    (x, y, w, h) = faces[0]

    # Extract the face ROI (region of interest)
    face_roi = gray_image[y:y+h, x:x+w]

    # Resize the face ROI to a fixed size
    face_roi = cv2.resize(face_roi, (200, 200))

    # Normalize the face ROI
    face_roi = face_roi / 255.0

    # Flatten the face ROI
    face_roi = face_roi.flatten()

    # Reshape the face ROI
    face_roi = face_roi.reshape(1, -1)

    # Name the face
    name = input("Enter a name for this face: ")
    # Save the face ROI and name to a file
    np.save("faces/" + name + ".npy", face_roi)

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    #frame = cv2.imread("image.jpg")
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Check if a face was detected
    print(len(faces))
    if len(faces) > 0:
        # Extract the coordinates of the first face
        for i in range(len(faces)):
            (x, y, w, h) = faces[i-1]

            # Extract the face ROI (region of interest)
            face_roi = gray_frame[y:y+h, x:x+w]

            # Resize the face ROI to a fixed size
            face_roi = cv2.resize(face_roi, (200, 200))

            # Normalize the face ROI
            face_roi = face_roi / 255.0

            # Flatten the face ROI
            face_roi = face_roi.flatten()

            # Reshape the face ROI
            face_roi = face_roi.reshape(1, -1)

            # Load the saved faces
            saved_faces = {}
            for file in os.listdir("faces"):
                name, _ = file.split(".")
                face = np.load("faces/" + file)
                saved_faces[name] = face

            # Initialize the minimum distance and name
            min_distance = 100
            name = None

            # Find the saved face with the minimum distance to the current face
            for key, value in saved_faces.items():
                distance = np.linalg.norm(face_roi - value)
                if distance < min_distance:
                    min_distance = distance
                    name = key

            # Draw a rectangle around the face and dqisplay the name
            print(min_distance)
            if min_distance < 100:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # else:
            #     name = input("Enter a name for this face: ")
            #     np.save("faces/" + name + ".npy", face_roi)


    # Show the frame with the detected and recognized face
    cv2.imshow("Webcam", frame)

    # Check if the user pressed "q" to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()