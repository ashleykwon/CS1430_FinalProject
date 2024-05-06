import socket
import cv2
import numpy as np
import struct


BUF_SIZE = 1280 * 720 * 2


def receive_video():
    # Set up socket
    HOST = '0.0.0.0'
    PORT = 9999
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    print("Waiting for a connection...")
    conn, addr = s.accept()
    print(f"Connected by {addr}")

    data = conn.recv(BUF_SIZE)
    payload_size = struct.calcsize("Q")

    w = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    h = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    c = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    msg_size = struct.unpack("Q", data[:payload_size])[0]
    data = data[payload_size:]

    try:
        while True:
            while len(data) < msg_size:
                data += conn.recv(BUF_SIZE)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape(w, h, c)

            # Process frame here

            # Send frame back to client

            cv2.imshow('Received', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        conn.close()
        s.close()

if __name__ == '__main__':
    receive_video()
