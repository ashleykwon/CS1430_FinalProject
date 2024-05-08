import cv2
import socket
import struct
import urllib.request
import numpy as np

BUF_SIZE = 1280 * 720 * 2
SERVER_URL = "http://0.0.0.0:5000" # This should be the same URL as the server's

def send_and_receive_video():
    HOST = '127.0.0.1'  # Replace with your server's IP address
    PORT = 5000

    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    # Start video capture
    cap = cv2.VideoCapture(0)

    # get frame size
    ret, frame = cap.read()
    w, h, c = frame.shape
    data = frame.flatten().tobytes()
    size = len(data)

    client_socket.sendall(struct.pack("Q", w) + struct.pack("Q", h) + struct.pack("Q", c) + struct.pack("Q", size))
   
    try:
        # Open the URL to receive video from
        # stream = urllib.reques.urlopen(SERVER_URL)

        while True:
            # Send video
            ret, frame = cap.read()
            if not ret:
                break
            data = frame.flatten().tobytes()
            client_socket.sendall(data)

            # Receive video from a url and stream the received video 
            # received_frame_bytes = stream.read(BUF_SIZE)
            # if not received_frame_bytes:
            #     break
            # received_frame = np.frombuffer(received_frame_bytes, dtype=np.uint8)
            # received_frame = cv2.imdecode(received_frame, 1)
            # cv2.imshow('Reconstructed', received_frame)
    finally:
        cap.release()
        client_socket.close()


if __name__ == '__main__':
    send_and_receive_video()
