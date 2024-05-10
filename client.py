import cv2
import socket
import struct
import urllib.request
import numpy as np

BUF_SIZE = 1280 * 720 * 2
HOST = '10.39.56.2'
PORT = 5000

def send_and_receive_video():
    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

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
    
    ## Capture video from one camera and send frames to the server
    try:
        while True:
            # Send video
            ret, frame = cap.read()
            if not ret:
                break
            data = frame.flatten().tobytes()
            client_socket.sendall(data)

            # Receive video from the server through socket
            received_data = client_socket.recv(BUF_SIZE)
            if received_data:
                # TODO: read received_data in the same format sent by multiClientServer and display it as a video
                print(received_data)
     
    finally:
        cap.release()
        client_socket.close()


if __name__ == '__main__':
    send_and_receive_video()
