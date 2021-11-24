import socket
from cv2 import cv2
import pickle
import struct
import sys


client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

PORT = 5001
host_ip = '127.0.0.1'

socket_addr = (host_ip, PORT)
client_socket.connect(socket_addr)

while True:
  cap = cv2.VideoCapture(0)
  try:
    while cap.isOpened():
      img, frame = cap.read()
      pickled_frame = pickle.dumps(frame)
      message = struct.pack("Q",len(pickled_frame))+pickled_frame
      client_socket.sendall(message)
      #cv2.imshow('client side', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  except Exception as msg:
    print("Client:Closing connection")
  finally:
    cap.release()
    cv2.destroyAllWindows()
    client_socket.close()
    sys.exit(0)


