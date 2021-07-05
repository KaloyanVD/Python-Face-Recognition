import cv2
from random import randrange
# https://youtu.be/XIrOM9oP3pA?t=1035


# the classifier detects the faces
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# choose an image to detect the faces
img = cv2.imread('two_faces.jpg')

webcam = cv2.VideoCapture(0)

while True:  # detects faces when using the default cam of the computer
    successful_frame_read, frame = webcam.read()  # reads the current frame
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draws rectangle for each frame
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256),
                                                  randrange(256), randrange(256)), 5)

    cv2.imshow('Front Face Detection', frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break
webcam.release()

# converting the image to grayscale so the algorithm works
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# drawing rectangle around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256),
                  randrange(256), randrange(256)), 5)

cv2.imshow('Front Face Detection', img)
cv2.waitKey()
