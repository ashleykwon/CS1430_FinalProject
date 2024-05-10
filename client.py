import cv2
import socket
import struct
import urllib.request
import numpy as np
from face_detection.face_detector import FaceDetector

BUF_SIZE = 1280 * 720 * 6
HOST = '10.39.56.2'
PORT = 5000


face_detector = FaceDetector('face_detection/detector_model.pb')
def detect_face(image):
    output_image = image.copy()

    input_array = np.asarray(image)[:, :, ::-1] # RGB
    faces, scores = face_detector(input_array, score_threshold=0.9)

    for (t, l, b, r) in faces:
        t, l, b, r = int(t), int(l), int(b), int(r)
        cv2.rectangle(output_image, (l, t), (r, b), (255,255,0), 2)

    if len(faces) != 0:
        return faces
    else:
        return []


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
    size = len(data)
    clientID = 1

    # Send frame metadata (width, height, channel, size, client ID) to the server ONCE
    client_socket.sendall(struct.pack("Q", w) + struct.pack("Q", h) + struct.pack("Q", c) + struct.pack("Q", size) + struct.pack("Q", clientID))
    
    received_data = b''

    ## Capture video from one camera and send frames to the server
    try:
        while True:
            # Send video
            ret, frame = cap.read()
            faceCoordinate = detect_face(frame)
            if not ret:
                break
            if len(faceCoordinate) != 0:
                print("face detected")
            # data = frame.flatten().tobytes()
                client_socket.sendall(faceCoordinate[0])

            # Receive video from the server through socket
            while len(received_data) < size:
                received_data += client_socket.recv(BUF_SIZE)

            rec_image_bytes = received_data[:size]
            received_data = received_data[size:]
            
            # Read received_data in the same format sent by multiClientServer and display it as a video
            if rec_image_bytes:
                rec_image = np.frombuffer(rec_image_bytes, dtype=np.uint8)
                rec_image = rec_image.reshape(w, h, c)
                cv2.imshow('Received', rec_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # print(received_data)
     
    finally:
        cap.release()
        client_socket.close()


if __name__ == '__main__':
    send_and_receive_video()
