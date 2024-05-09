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
    cap1 = cv2.VideoCapture(1) # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA

    # get frame size and generate client ID
    ret, frame = cap0.read()
    ret1, frame1 = cap1.read() # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
    w, h, c = frame.shape
    w1, h1, c1 = frame1.shape # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
    data = frame.flatten().tobytes() # frame from camera 1
    data2 = frame1.flatten().tobytes() # frame from camera 2 # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
    size = len(data)+len(data2) # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
    # size = len(data) # UNCOMMENT THIS FOR DEBUGGING
    clientID = 2

    # Send the videos captured from two webcams to the server
    client_socket.sendall(struct.pack("Q", w+w1) + struct.pack("Q", h+h1) + struct.pack("Q", c+c1) + struct.pack("Q", size)) # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
    # client_socket.sendall(struct.pack("Q", w) + struct.pack("Q", h) + struct.pack("Q", c) + struct.pack("Q", size) + struct.pack("Q", clientID)) # UNCOMMENT THIS FOR DEBUGGING
    
    # send video
    try:
        while True:
            ret, frame = cap0.read()
            # ret1, frame1 = cap1.read()
            if not ret:
                break
            joined_frame = np.concatenate((frame, frame1)) # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
            joined_frame_data = joined_frame.flatten().tobytes() # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
            client_socket.sendall(joined_frame_data) # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
            # client_socket.sendall(frame) # UNCOMMENT THIS FOR DEBUGGING
    finally:
        cap0.release()
        cap1.release() # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
        client_socket.close()


if __name__ == '__main__':
    send_video()
