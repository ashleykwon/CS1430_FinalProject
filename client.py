import cv2
import socket
import struct

def send_video():
    HOST = '0.0.0.0'  # Replace with your server's IP address
    PORT = 9999

    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    # Start video capture
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            data = frame.flatten().tobytes()
            size = len(data)
            client_socket.sendall(struct.pack("Q", size) + data)  # Prefix each message with a length (Q = unsigned long long)
    finally:
        cap.release()
        client_socket.close()

if __name__ == '__main__':
    send_video()
