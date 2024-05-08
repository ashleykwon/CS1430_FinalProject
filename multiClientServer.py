from _thread import *
import socket
import sys
import struct
import cv2
import numpy as np
import urllib.request

# Referenced from https://stackoverflow.com/questions/10810249/python-socket-multiple-clients

BUF_SIZE = 1280 * 720 * 2
URL = "http://0.0.0.0:5000" # Change this to the IP address of the computer this program is running

def clientthread(conn):
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

    # For debugging purposes only! 
    # capDebugging = cv2.VideoCapture(0)

    # # get frame size
    # ret, frame = capDebugging.read()
    # w, h, c = frame.shape
    # data = frame.flatten().tobytes()
    # size = len(data)

    try:
        while True:
            while len(data) < msg_size:
                data += conn.recv(BUF_SIZE)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            # frame = np.frombuffer(frame_data, dtype=np.uint8)
            # frame = frame.reshape(w, h, c)

            # Process frame here
            # processedFrame = np.zeros((10, 10)) # REPLACE THIS TO A REAL FRAME
            
            # Send frame back to client by posting it to a url 
            # stream = urllib.request.urlopen(URL)
            # processedFrame.open(stream)
            
            # cv2.imshow('Received', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    finally:
        conn.close()

def main():
    try:
        host = '127.0.0.1'
        port = 5000
        tot_socket = 2
        list_sock = []
        for i in range(tot_socket):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            s.bind((host, port+i))
            s.listen(10)
            list_sock.append(s)
            print("[*] Server listening on %s %d" %(host, (port+i)))

        while True:
            for j in range(len(list_sock)):
                conn, addr = list_sock[j].accept()
                print('[*] Connected with ' + addr[0] + ':' + str(addr[1])) # 
                start_new_thread(clientthread ,(conn,))
        s.close()
    except KeyboardInterrupt as msg:
        sys.exit(0)


if __name__ == "__main__":
    main()