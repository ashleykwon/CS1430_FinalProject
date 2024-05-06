import cv2
import socket
import pickle
import struct

# Server configuration
SERVER_HOST = '172.16.12.11'  # Listen on all available interfaces
SERVER_PORT = 3000

# Socket creation
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(5)

print("[INFO] Server started, waiting for clients...")

# Accept a single connection
client_socket, client_address = server_socket.accept()
print(f"[INFO] Client connected: {client_address}")

# Open default camera (index 0)
cap = cv2.VideoCapture(0) 
# add another video capture like the one above with a different ID for the two webcam setup

try:
    while True:
        # Capture frame from camera
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) # for mirroring
        # width  = cap.get(3) #1280
        # height = cap.get(4) #720
        # print(width)
        # print(height)

        # Serialize frame
        data = pickle.dumps(frame)
        
        # Send frame size
        message_size = struct.pack("L", len(data))
        client_socket.sendall(message_size + data)
    
except KeyboardInterrupt:
    print("[INFO] Keyboard interrupt received, closing connection...")

finally:
    # Release the camera and close connection
    cap.release()
    client_socket.close()
    server_socket.close()