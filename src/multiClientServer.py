from _thread import *
import socket
import struct
import cv2
import numpy as np
import threading
from projection.zoe_depth import ZoeDepth
from projection.camera import get_intrinsic_matrix
from projection.pipeline import reprojectImages
import torch
from PIL import Image
import pickle

# Referenced from:
#   https://stackoverflow.com/questions/10810249/python-socket-multiple-clients
#   https://realpython.com/intro-to-python-threading/
 
BUF_SIZE = 1280 * 720 * 2
HOST = "10.39.56.2"
PORT = 5000

zoe_depth = ZoeDepth(device=("cuda" if torch.cuda.is_available() else "cpu"))

def client_thread_function(client_socket):
    global faceCoordinate  # video frame from 1 camera from client 1 for face detection
    global dataFor2Dto3D  # video frames from 2 cameras from client 2 for 3D reconstruction
    global dataFor3Dto2D  # video frame where the 3D reconstruction result is turned into 2D to be sent back to client 1
    global calibration_size # calibration matrix size from client 2
    global calibrationMatrices # list with calibration matrices from client 2
    data = client_socket.recv(BUF_SIZE)
    payload_size = struct.calcsize("Q")

    w_2 = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    h = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    c = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    msg_size = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    received_clientID = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    # TODO:
    # if client 2
    if received_clientID == 2:
        # Load calibration matrix size
        calibration_size = struct.unpack("Q", data[:payload_size])[0]
        data = data[payload_size:]
        # Load calibration matrices
        calibrationMatrices = list(pickle.loads(data[:calibration_size]))
        data = data[calibration_size:]
    else:
        calibration_size = 0
        calibrationMatrices = []
    # get calibration matrices
    # K_l, R_l, t_l, K_r, R_r, t_r = pickle.loads(incoming_bytes)

    try:
        while True:
            # Receive data from client sockets
            while len(data) < msg_size:
                data += client_socket.recv(BUF_SIZE)
            if received_clientID == 1:  # data for face detection received from client 1
                faceCoordinate = data[:msg_size]
                if faceCoordinate:
                    faceCoordinate = list(pickle.loads(faceCoordinate))
                else:
                    faceCoordinate = [w//2, h//2] # if no face was detected, set faceCoordinate to the center of the client1 camera frame
                dataFor2Dto3D = b""
            elif (
                received_clientID == 2
            ):  # data for 2D to 3D reconstruction received from client 2
                dataFor2Dto3D = data[:msg_size]
            data = data[msg_size:]
            

            # Visualize received frame for debugging porposes only
            # frame = np.frombuffer(dataForFD, dtype=np.uint8)
            # frame = frame.reshape(w, h, c)
            # cv2.imshow('Received', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break


            if dataFor2Dto3D != b"":
                joined_frames = np.frombuffer(dataFor2Dto3D, dtype=np.uint8).reshape(w_2, h, c)
                w = w_2 // 2

                leftCameraFrame = Image.fromarray(
                    cv2.cvtColor(joined_frames[:w, :, :], cv2.COLOR_BGR2RGB)
                )

                rightCameraFrame = Image.fromarray(
                    cv2.cvtColor(joined_frames[w:, :, :], cv2.COLOR_BGR2RGB)
                )

                # Calibration matrices for the left camera
                K_l = calibrationMatrices[0]
                R_l = calibrationMatrices[1]
                t_l = calibrationMatrices[2]
                
                # Calibration matrices for the right camera
                K_r = calibrationMatrices[3]
                R_r = calibrationMatrices[4]
                t_r = calibrationMatrices[5]

                new_x = faceCoordinate[0]
                new_y = faceCoordinate[1]

                reprojected_image = reprojectImages(leftCameraFrame, rightCameraFrame, zoe_depth, K_l, R_l, t_l, K_r, R_r, t_r, new_x, new_y)
                # reprojected_image = cv2.cvtColor(reprojected_image, cv2.COLOR_RGB2BGR)

                # TODO: send reprojected image back

                # Send the 3D to 2D mapping result back to client 1
                if received_clientID == 1:  # from client 1
                    dataFor3Dto2D = reprojected_image.flatten().tobytes()
                    client_socket.sendall(dataFor3Dto2D)

    finally:
        client_socket.close()


def main():
    # Create a server socket and bind it to an address/port
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))

    # Listen for incoming connections
    server_socket.listen(5)

    try:
        while True:
            # Accept a connection (Should accept the connection from client 1 before client 2)
            client_socket, client_address = server_socket.accept()
            print(f"Got connection from {client_address}")

            # Create a thread to handle the client
            client_thread = threading.Thread(
                target=client_thread_function, args=(client_socket,)
            )
            client_thread.start()

    except KeyboardInterrupt:
        print("Server shutting down.")


if __name__ == "__main__":
    main()
