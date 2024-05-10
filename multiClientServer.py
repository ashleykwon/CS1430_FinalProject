from _thread import *
import socket
import sys
import struct
import cv2
import numpy as np
import urllib.request
import threading


# face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

# def detect_face(image):
#     output_image = image.copy()
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(image_gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     return output_image

# Referenced from https://stackoverflow.com/questions/10810249/python-socket-multiple-clients

BUF_SIZE = 1280 * 720 * 2
HOST = '10.39.56.2'
PORT = 5000

def clientthread(client_socket, client_id, clients):
    global dataForFD # video frame from 1 camera from client 1 for face detection 
    global dataFor2Dto3D # video frames from 2 cameras from client 2 for 3D reconstruction
    global dataFor3Dto2D # video frame where the 3D reconstruction result is turned into 2D to be sent back to client 1

    data = client_socket.recv(BUF_SIZE)
    payload_size = struct.calcsize("Q")

    w = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    h = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    c = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    msg_size = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    received_clientID = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    try:
        while True:
            # Receive data from client sockets
            while len(data) < msg_size:
                data += client_socket.recv(BUF_SIZE)
            if received_clientID == 1: # data for face detection received from client 1
                dataForFD = data[:msg_size]
            elif received_clientID == 2: # data for 2D to 3D reconstruction received from client 2
                dataFor2Dto3D = data[:msg_size] 
            data = data[msg_size:]

            print(dataForFD)
            
            # frame = np.frombuffer(dataForFD, dtype=np.uint8)
            # frame = frame.reshape(w, h, c)
            # cv2.imshow('Received', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            # TODO 1: Do the face detection on faceDetectionInput

            # faceDetectionInput = np.frombuffer(dataForFD, dtype=np.uint8)
            # faceDetectionInput = faceDetectionInput.reshape(w, h, c) #this changes dataForFD into a numpy array with size (w, h, c)
            # dataFor3Dto2D = detect_face(faceDetectionInput)
            
            # dataFor3Dto2D = faceDetectionInput

            # TODO 2: Do the 2D to 3D reconstruction on dataFor2Dto3D

            # TODO 3: Do the 3D to 2D mapping + viewing angle modification based on face detection and save the result in dataFor3Dto2D
            # dataFor3Dto2D = b'sample output' # Change this to the actual output to client 1

            # FOR DEBUGGING PURPOSES ONLY: Check if dataForFD is a frame from the video captured by client 1
            # frame = np.frombuffer(dataForFD, dtype=np.uint8)
            # print(frame.shape)
            # frame = frame.reshape(w, h, c)
            # cv2.imwrite('Received.png', frame) 

            # Send the 3D to 2D mapping result back to client 1
            if received_clientID == 1: # from client 1
                # dataFor3Dto2D = dataFor3Dto2D.flatten().tobytes()
                dataFor3Dto2D = dataForFD
                client_socket.sendall(dataFor3Dto2D)
          
    finally:
        client_socket.close()

   



def main():
    # Create a server socket and bind it to the address/port
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    
    # Listen for incoming connections 
    server_socket.listen(2)

    # Dictionary to store connected clients
    clients = {}

    # Counter for client IDs
    client_id_counter = 1

    try:
        # while True:
        # Accept a connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")

        # Assign a unique ID to the client
        client_id = client_id_counter
        client_id_counter += 1

        # Add the client to the dictionary
        clients[client_id] = client_socket

        # Create a thread to handle the client
        client_thread = threading.Thread(target=clientthread, args=(client_socket, client_id, clients))
        client_thread.start()

    except KeyboardInterrupt:
        print("Server shutting down.")
        


if __name__ == "__main__":
    main()