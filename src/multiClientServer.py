from _thread import *
import socket
import struct
import cv2
import numpy as np
import threading
from projection.geometry import depth_to_points
from projection.zoe_depth import ZoeDepth
from projection.camera import get_intrinsic_matrix
import torch
from PIL import Image

# Referenced from https://stackoverflow.com/questions/10810249/python-socket-multiple-clients

BUF_SIZE = 1280 * 720 * 2
HOST = "10.39.56.2"
PORT = 5000

zoe_depth = ZoeDepth(device=("cuda" if torch.cuda.is_available() else "cpu"))

# dataFor2Dto3D = b''


def clientthread(client_socket, client_id, clients):
    global faceCoordinate  # video frame from 1 camera from client 1 for face detection
    global dataFor2Dto3D  # video frames from 2 cameras from client 2 for 3D reconstruction
    global dataFor3Dto2D  # video frame where the 3D reconstruction result is turned into 2D to be sent back to client 1

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
            if received_clientID == 1:  # data for face detection received from client 1
                faceCoordinate = data[:msg_size]
                dataFor2Dto3D = b""
            elif (
                received_clientID == 2
            ):  # data for 2D to 3D reconstruction received from client 2
                dataFor2Dto3D = data[:msg_size]
            data = data[msg_size:]
            # print(dataFor2Dto3D)
            # Print the user's face position (2D coordinate) for debugging purposes
            # print(pickle.loads(faceCoordinate))

            # Visualize received frame for debugging porposes only
            # frame = np.frombuffer(dataForFD, dtype=np.uint8)
            # frame = frame.reshape(w, h, c)
            # cv2.imshow('Received', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # TODO 2: Do the 2D to 3D reconstruction on dataFor2Dto3D
            if dataFor2Dto3D != b"":
                joined_frames = np.frombuffer(dataFor2Dto3D, dtype=np.uint8)
                joined_frames = joined_frames.reshape(
                    w, h, c
                )  # this changes dataForFD into a numpy array with size (w, h, c)
                singleFrameWidth = int(
                    w // 2
                )  # Assumes frames from the two video cameras are joined side by side

                leftCameraFrame, rightCameraFrame = (
                    joined_frames[:singleFrameWidth, :, :],
                    joined_frames[singleFrameWidth:, :, :],
                )
                leftCameraFrame_pil = Image.fromarray(
                    cv2.cvtColor(leftCameraFrame, cv2.COLOR_BGR2RGB)
                )
                rightCameraFrame_pil = Image.fromarray(
                    cv2.cvtColor(rightCameraFrame, cv2.COLOR_BGR2RGB)
                )

                # print(leftCameraFrame.shape)
                # print(rightCameraFrame.shape)
                dataFor3Dto2D = rightCameraFrame

                intrinsicMatrix = get_intrinsic_matrix(
                    fov_x=82.1, fov_y=52.2, W=1920, H=1080
                )  # Should be the same across the two webcams and the client1's head (aka third camera)
                leftCameraRotation = np.asarray(
                    [
                        [0.9117811489826978, 0.07599962415662805, 0.4035829449912901],
                        [
                            -0.06867995442145704,
                            0.9971058203375325,
                            -0.03260440015830532,
                        ],
                        [
                            -0.40489282559766104,
                            0.002010019170952072,
                            0.9143619412478159,
                        ],
                    ]
                )
                leftCameraTranslation = np.asarray(
                    [-19.269844497623666, 1.1276310094741875, 5.48936711837891]
                )

                # TODO: can we do this in one pass?
                leftCameraDepth = zoe_depth.get_depth(leftCameraFrame_pil)
                rightCameraDepth = zoe_depth.get_depth(rightCameraFrame_pil)

                leftCameraTo3D = depth_to_points(
                    leftCameraDepth,
                    intrinsicMatrix,
                    leftCameraRotation,
                    leftCameraTranslation,
                )
                rightCameraTo3D = depth_to_points(
                    rightCameraDepth, intrinsicMatrix, np.eye(3), np.zeros(3)
                )

                # print(leftCameraTo3D.shape)
                # print(rightCameraTo3D.shape)

                # TODO 3: Do the 3D to 2D mapping + viewing angle modification based on face detection and save the result in dataFor3Dto2D
                # dataFor3Dto2D = b'sample output' # Change this to the actual output to client 1
                # dataFor3Dto2D SHOULD BE A NUMPY ARRAY

                # translation = np.multiply(leftCameraTranslation, np.asarray([faceCoordinate[0], faceCoordinate[1], 0]))
                # rotation = np.zeros(3, 3)

                # extrinsicMatrix = np.hstack((rotation, translation))

                # Use the extrinsic and the intrinsic matrices to get uv coordinates
                dataFor3Dto2D = rightCameraFrame  # change this to an actual output

                # FOR DEBUGGING PURPOSES ONLY: Check if dataForFD is a frame from the video captured by client 1
                # frame = np.frombuffer(dataForFD, dtype=np.uint8)
                # print(frame.shape)
                # frame = frame.reshape(w, h, c)
                # cv2.imwrite('Received.png', frame)
                # dataFor3Dto2D = rightCameraFrame # change this to the actual output

                # Send the 3D to 2D mapping result back to client 1
                if received_clientID == 1:  # from client 1
                    dataFor3Dto2D = dataFor3Dto2D.flatten().tobytes()
                    client_socket.sendall(dataFor3Dto2D)

    finally:
        client_socket.close()


def main():
    # Create a server socket and bind it to the address/port
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))

    # Listen for incoming connections
    server_socket.listen(10)

    # Dictionary to store connected clients
    clients = {}

    # Counter for client IDs
    client_id_counter = 1

    try:
        while True:
            # Accept a connection
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")

            # Assign a unique ID to the client
            client_id = client_id_counter
            client_id_counter += 1

            # Add the client to the dictionary
            clients[client_id] = client_socket

            # Create a thread to handle the client
            client_thread = threading.Thread(
                target=clientthread, args=(client_socket, client_id, clients)
            )
            client_thread.start()

    except KeyboardInterrupt:
        print("Server shutting down.")


if __name__ == "__main__":
    main()
