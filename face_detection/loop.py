import cv2
from face_detector import FaceDetector
import numpy as np

face_detector = FaceDetector('detector_model.pb')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    input_array = np.asarray(img)[:, :, ::-1] # RGB
    boxes, scores = face_detector(input_array, score_threshold=0.9)

    print(boxes)
    # exit()

    for (t, l, b, r) in boxes:
        t, l, b, r = int(t), int(l), int(b), int(r)
        cv2.rectangle(img,(l, t),(r, b),(255,255,0),2)

    # Display an image in a window
    cv2.imshow('img',img)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
