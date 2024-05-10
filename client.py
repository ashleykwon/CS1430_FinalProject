import cv2
import socket
import struct
import urllib.request
import numpy as np
import pickle
from face_detection.face_detector import FaceDetector

BUF_SIZE = 1280 * 720 * 6
HOST = '10.39.56.2'
PORT = 5000


face_detector = FaceDetector('face_detection/detector_model.pb')
def detect_face(image):
    input_array = np.asarray(image)[:, :, ::-1] # RGB
    faces, scores = face_detector(input_array, score_threshold=0.9)

    if len(faces) == 0:
        return None
    else:
        t, l, b, r = faces[np.argmax(scores)]
        c_x, c_y = (l + r) // 2, (t + b) // 2
        height, width, channel = image.shape
        # print(width) # 720
        # print(height) # 1280
        return (c_x/width, c_y/height)


def send_and_receive_video():
    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    print("connected")

    # Start video capture
    cap = cv2.VideoCapture(0)

    # get frame size and generate client ID
    ret, frame = cap.read()
    w, h, c = frame.shape
    data = frame.flatten().tobytes()
    frameSize = len(data)

    faceCoordinates = detect_face(frame)
    face_bytes = pickle.dumps(faceCoordinates)
    size = len(face_bytes)
    clientID = 1

    # Send frame metadata (width, height, channel, size, client ID) to the server ONCE
    client_socket.sendall(struct.pack("Q", w) + struct.pack("Q", h) + struct.pack("Q", c) + struct.pack("Q", size) + struct.pack("Q", clientID))
    
    received_data = b''

    ## Capture video from one camera, detect face, and send face coordinate (x, y) to the server
    try:
        while True:
            ret, frame = cap.read()
            faceCoordinates = detect_face(frame)
            
            if faceCoordinates is not None:
                face_bytes = pickle.dumps(faceCoordinates)
                client_socket.sendall(face_bytes)
            if not ret:
                break
            

            # Receive video from the server through socket
            # while len(received_data) < frameSize:
            #     print("Stuck!")
            #     received_data += client_socket.recv(BUF_SIZE)
            # rec_image_bytes = received_data[:frameSize]
            # received_data = received_data[frameSize:]
            
            # # Read received_data in the same format sent by multiClientServer and display it as a video
            # if rec_image_bytes:
            #     rec_image = np.frombuffer(rec_image_bytes, dtype=np.uint8)
            #     rec_image = rec_image.reshape(w, h, c)
            #     cv2.imshow('Received', rec_image)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
                
                # print(received_data)
     
    finally:
        cap.release()
        client_socket.close()


if __name__ == '__main__':
    send_and_receive_video()
