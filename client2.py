import cv2
import socket
import struct
import numpy as np

def send_video():
    HOST = '127.0.0.1'  # Replace with your server's IP address
    PORT = 5000

    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((HOST, PORT))
    except socket.error as e:
        print(f"Error connecting to {HOST}:{PORT}: {e}")

    # Start video capture using the two webcams
    cap0 = cv2.VideoCapture(0)
    # cap1 = cv2.VideoCapture(1)

    # get frame size and generate client ID
    ret, frame = cap0.read()
    # ret1, frame1 = cap1.read()
    w, h, c = frame.shape
    # w1, h1, c1 = frame1.shape
    data = frame.flatten().tobytes() # frame from camera 1
    # data2 = frame1.flatten().tobytes() # frame from camera 2
    # size = len(data)+len(data2)
    size = len(data) # for debugging purposes only 
    clientID = 2

    # Send the videos captured from two webcams to the server
    # client_socket.sendall(struct.pack("Q", w+w1) + struct.pack("Q", h+h1) + struct.pack("Q", c+c1) + struct.pack("Q", size))
    client_socket.sendall(struct.pack("Q", w) + struct.pack("Q", h) + struct.pack("Q", c) + struct.pack("Q", size) + struct.pack("Q", clientID)) 
    # send video
    try:
        while True:
            ret, frame = cap0.read()
            # ret1, frame1 = cap1.read()
            if not ret:
                break
            # joined_frame = np.concatenate((frame, frame1))
            # joined_frame_data = joined_frame.flatten().tobytes()
            # client_socket.sendall(joined_frame_data)
            client_socket.sendall(frame) # for debugging purposes only. Use the line above instead 
    finally:
        cap0.release()
        # cap1.release()
        client_socket.close()


if __name__ == '__main__':
    send_video()
