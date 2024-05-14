import cv2
import socket
import struct
import numpy as np
import tyro
import pickle


def prepare_initial_payload(frame, frame1, left_calibration_file, right_calibration_file):
    # get frame size and generate client ID
     # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
    h, w, c = frame.shape
    h1, w1, c1 = frame1.shape  # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
    # data = frame.flatten().tobytes()  # frame from camera 1
    data = pickle.dumps(np.hstack((frame, frame1)))
    # data2 = (
    #     frame1.flatten().tobytes()
    # )  # frame from camera 2 # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
    size = len(data)  # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
    clientID = 2

    K_l, dist_l, R_l, t_l = pickle.load(open(left_calibration_file, "rb"))
    K_r, dist_r, R_r, t_r = pickle.load(open(right_calibration_file, "rb"))

    calibration_bytes = pickle.dumps((K_l, dist_l, R_l, t_l, K_r, dist_r, R_r, t_r))
    calibration_size = len(calibration_bytes)

    payload_bytes = (
        struct.pack("Q", w + w1)
        + struct.pack("Q", h)
        + struct.pack("Q", c)
        + struct.pack("Q", size)
        + struct.pack("Q", clientID)
        + struct.pack("Q", calibration_size)
        + calibration_bytes
    )

    return payload_bytes


def send_video(
    host: str = "10.39.56.2",
    port: int = 5000,
    left_calibration_file: str = "test_data/left_camera.pickle",
    right_calibration_file: str = "test_data/right_camera.pickle",
):
    # Start video capture using the two webcams
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)  # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA

    ret, frame = cap0.read()
    ret1, frame1 = cap1.read() 

    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
    except socket.error as e:
        print(f"Error connecting to {host}:{port}: {e}")

    payload_bytes = prepare_initial_payload(frame, frame1, left_calibration_file, right_calibration_file)
    client_socket.sendall(payload_bytes)  # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA

    # client_socket.sendall(struct.pack("Q", w) + struct.pack("Q", h) + struct.pack("Q", c) + struct.pack("Q", size) + struct.pack("Q", clientID) + struct.pack("Q", calibration_size) + calibration_bytes) # UNCOMMENT THIS FOR DEBUGGING

    # send video
    try:
        while True:
            ret, frame = cap0.read()
            ret1, frame1 = cap1.read()  # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
            if not ret:
                break
            joined_frame = np.hstack(
                (frame, frame1)
            )  # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA

            # cv2.imwrite("joined frame.jpg", joined_frame) # This is fine
            joined_frame_data = pickle.dumps(joined_frame)  # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
            # print("joined frame shape in client 2" + str(joined_frame.shape))
            client_socket.sendall(
                joined_frame_data
            )  # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
            # client_socket.sendall(frame) # UNCOMMENT THIS FOR DEBUGGING
    finally:
        cap0.release()
        cap1.release()  # COMMENT THIS OUT WHEN ONLY USING ONE CAMERA
        client_socket.close()


if __name__ == "__main__":
    tyro.cli(send_video)
