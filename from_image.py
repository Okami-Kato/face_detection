from imutils import face_utils
import cv2
import dlib
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--predictor", required=True,
                help="path to trained dlib shape predictor model")
ap.add_argument("-i", "--image", required=True,
                help="path to image")
args = vars(ap.parse_args())

original_image = cv2.imread(args["image"])
resized_image = imutils.resize(original_image, 600)
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
predictor = dlib.shape_predictor(args["predictor"])

rects = face_cascade.detectMultiScale(gray)
for (x, y, w, h) in rects:
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    shape = predictor(resized_image, dlib.rectangle(x, y, x + w, y + h))
    shapeNp = face_utils.shape_to_np(shape)

for (sX, sY) in shapeNp:
    cv2.circle(resized_image, (sX, sY), 1, (0, 0, 255), -1)

cv2.imshow("Frame", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
