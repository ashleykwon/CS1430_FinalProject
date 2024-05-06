import cv2
import socket
import pickle
import struct

# Server configuration
SERVER_HOST = '172.16.12.11'  # Change to the IP address of the server
SERVER_PORT = 3000

# Socket creation
try:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.settimeout(10)
    client_socket.connect((SERVER_HOST, SERVER_PORT))
except socket.error as e:
    print(f"Error connecting to {SERVER_HOST}:{SERVER_PORT}: {e}")

# Data receiving loop
data = b''
payload_size = struct.calcsize("L")

while True:
    # print("data received")
    # Receive frame size
    while len(data) < payload_size:
        packet = client_socket.recv(1280*720)
        if not packet: break
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    # Receive frame data
    while len(data) < msg_size:
        data += client_socket.recv(1280*720)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Deserialize frame
    frame = pickle.loads(frame_data)

    # Display the received frame
    cv2.imshow('Received', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close connection and cleanup
client_socket.close()
cv2.destroyAllWindows()