import cv2
import socket
import struct
import urllib.request
import numpy as np

BUF_SIZE = 1280 * 720 * 2

def send_and_receive_video():
    HOST = '127.0.0.1'  # Replace with your server's IP address
    PORT = 5000

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

    client_socket.sendall(struct.pack("Q", w) + struct.pack("Q", h) + struct.pack("Q", c) + struct.pack("Q", size) + struct.pack("Q", clientID))
    
    ## ALWAYS RUN CLIENT 1 BEFORE RUNNING CLIENT 2 
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
                print(received_data)
                # Display the received frame
     
    finally:
        cap.release()
        client_socket.close()


if __name__ == '__main__':
    send_and_receive_video()
