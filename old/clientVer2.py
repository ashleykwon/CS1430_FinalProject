import cv2
import socket
import pickle
import struct
import numpy as np

# Server configuration
SERVER_HOST = '127.0.0.1'  # Listen on all available interfaces
SERVER_PORT = 3000

# Socket creation
try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.settimeout(10)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
except socket.error as e:
    print(f"Error connecting to {SERVER_HOST}:{SERVER_PORT}: {e}")

# Open default camera (index 0)
cap = cv2.VideoCapture(0) 
cap1 = cv2.VideoCapture(1) 
# cap = cv2.VideoCapture(0) 
# add another video capture like the one above with a different ID for the two webcam setup

try:
    while True:
        # Capture frame from camera
        ret, frame = cap.read()
        ret1, frame1 = cap1.read()
        joined_frame = np.concatenate((frame, frame1))

        # joined_frame = cv2.flip(joined_frame, 1) # for mirroring
        # width  = cap.get(3) #1280
        # height = cap.get(4) #720
        # print(width)
        # print(height)

        data = pickle.dumps(joined_frame)
        
        # Send frame size
        message_size = struct.pack("L", len(data))
        client_socket.sendall(message_size + data)
    
except KeyboardInterrupt:
    print("[INFO] Keyboard interrupt received, closing connection...")

finally:
    # Release the camera and close connection
    cap.release()
    client_socket.close()
    