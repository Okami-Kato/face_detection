from imutils.video import VideoStream
from imutils import face_utils
import argparse
import cv2
import dlib
import imutils
import time

# cap = cv2.VideoCapture(0)
# original_image = cv2.imread('resources/20220508_143630.jpg')

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--predictor", required=True,
                help="path to trained dlib shape predictor model")
args = vars(ap.parse_args())

vs = VideoStream(0).start()
time.sleep(2.0)

face_cascade = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
predictor = dlib.shape_predictor(args["predictor"])

while True:
    frame = vs.read()
    frame = imutils.resize(frame, 600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        shape = predictor(gray, dlib.rectangle(x, y, x + w, y + h))
        shapeNp = face_utils.shape_to_np(shape)

        for (sX, sY) in shapeNp:
            cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
