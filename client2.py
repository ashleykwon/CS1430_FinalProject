import cv2
import socket
import struct
import numpy as np

def send_video():
    HOST = '127.0.0.1'  # Replace with your server's IP address
    PORT = 5001

    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    # Start video capture
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    # get frame size
    ret, frame = cap0.read()
    ret1, frame1 = cap1.read()
    w, h, c = frame.shape
    w1, h1, c1 = frame1.shape
    data = frame.flatten().tobytes() # frame from camera 1
    data2 = frame1.flatten().tobytes() # frame from camera 2
    size = len(data)+len(data2)

    client_socket.sendall(struct.pack("Q", w+w1) + struct.pack("Q", h+h1) + struct.pack("Q", c+c1) + struct.pack("Q", size))

    # send video
    try:
        while True:
            ret, frame = cap0.read()
            ret1, frame1 = cap1.read()
            if not ret:
                break
            joined_frame = np.concatenate((frame, frame1))
            joined_frame_data = joined_frame.flatten().tobytes()
            client_socket.sendall(joined_frame_data)
    finally:
        cap0.release()
        cap1.release()
        client_socket.close()


if __name__ == '__main__':
    send_video()
