import cv2
import socket
import pickle
import struct

# Server configuration
SERVER_HOST = '127.0.0.1'  # Change to the IP address of the server
SERVER_PORT = 3000

# Socket creation
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_HOST, SERVER_PORT))
server_socket.listen(5)

print("[INFO] Server started, waiting for clients...")

# Accept a single connection
client_socket, client_address = server_socket.accept()
print(f"[INFO] Client connected: {client_address}")


# Data receiving loop
data = b''
payload_size = struct.calcsize("L")

while True:
    # print("data received")
    # Receive frame size
    while len(data) < payload_size:
        packet = client_socket.recv(1280*720*2)
        if not packet: break
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    # Receive frame data
    while len(data) < msg_size:
        data += client_socket.recv(1280*720*2)
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
server_socket.close()
cv2.destroyAllWindows()

