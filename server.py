import pickle
import struct
import socket
import cv2
from datetime import datetime
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

def log(conn, addr):
  stamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
  addr = ':'.join(list(map(lambda x: str(x), list(addr))))
  log_in = "[%s]:[%s]:%s" % (stamp, addr, conn)
  with open("log.txt", "a") as log_file:
    log_file.write(log_in + '\n')
  print(log_in)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

PORT = 5001
host_ip = '192.168.0.143'

host_ip = '127.0.0.1'
socket_addr = (host_ip, PORT)
server_socket.bind(socket_addr)

server_socket.listen(5)

print("Listening on port: %s"%(socket_addr, ))

while True:
  client_socket, addr = server_socket.accept()
  try:
    log("Client Connected", addr)
    # get client data
    data = b""
    payload_size = struct.calcsize("Q")
    while True:
      while len(data) < payload_size:
        packet = client_socket.recv(4*1024)
        if not packet:
          break
        data += packet
      packed_msg_size = data[ :payload_size]
      data = data[payload_size: ]
      msg_size = struct.unpack("Q", packed_msg_size)[0]
      while (len(data) < msg_size):
        data += client_socket.recv(4*1024)
      frame_data = data[ :msg_size]
      data = data[msg_size: ]
      frame = pickle.loads(frame_data)
      cv2.imshow("Server", frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(1)
  except struct.error:
    print("Server:Client sent 0 bytes")
    client_socket.close()
    cv2.destroyAllWindows()
    continue
  except socket.error:
    print("Server:Client closed Socket")
    client_socket.close()
    cv2.destroyAllWindows()
    continue
  except:
    break
  finally:
    log("Connection Closed", addr)


