import socket
import cv2
import numpy as np
import struct

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

    # Receive data
    data = b''
    payload_size = struct.calcsize("Q")  # Unsigned long long
    try:
        while True:
            while len(data) < payload_size:
                data += conn.recv(4096)
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            # Extract frame
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape(1080, 1920, 3)  # TODO: encode shape in message
            cv2.imshow('Received', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        conn.close()
        s.close()

if __name__ == '__main__':
    receive_video()
