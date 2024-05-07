import cv2
import socket
import struct

def send_video():
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

    # send video

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            data = frame.flatten().tobytes()
            client_socket.sendall(data)
    finally:
        cap.release()
        client_socket.close()


if __name__ == '__main__':
    send_video()
