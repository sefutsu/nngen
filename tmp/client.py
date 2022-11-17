import sys
import socket
import numpy as np

encoding = "utf-8"

class BaseClient:
    def __init__(self, timeout:int=10, buffer:int=1024):
        self._socket = None
        self._address = None
        self._timeout = timeout
        self._buffer = buffer

    def connect(self, address, family:int, typ:int, proto:int):
        self._address = address
        self._socket = socket.socket(family, typ, proto)
        self._socket.settimeout(self._timeout)
        self._socket.connect(self._address)
        print("Connect to", address)

    def send(self, message:str="") -> None:
        flag = False
        while True:
            if message == "":
                message_send = input("> ")
            else:
                message_send=message
                flag = True
            self._socket.send(message_send.encode(encoding))
            self.receive()
            if flag:
                break
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
            self._socket.close()
        except:
            pass

    def receive(self):
        message = self._socket.recv(self._buffer).decode(encoding)
        print(message)

class PYNQClient(BaseClient):
    def __init__(self, host:str="192.168.3.1", port:int=8010) -> None:
        self.server=(host,port)
        super().__init__(timeout=60, buffer=1024)
        super().connect(self.server, socket.AF_INET, socket.SOCK_STREAM, 0)

    def run(self):
        self._socket.send("run".encode(encoding))
        self.receive()

    def set_buffer(self, idx_begin, idx_end, values):
        if idx_end - idx_begin != len(values):
            print("Invalid length", file=sys.stderr)
            return
        self._socket.send(f"setbuf {idx_begin} {idx_end}".encode(encoding))
        self._socket.send(values.tobytes())
        self.receive()

    def read_buffer(self, idx_begin, idx_end):
        self._socket.send(f"readbuf {idx_begin} {idx_end}".encode(encoding))

if __name__ == '__main__':
    port = 8010
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    client = PYNQClient(port=port)

    a = np.array([2, 3, 4, 5, 6], dtype=np.int8)
    client.set_buffer(1, 6, a)
    # client.read_buffer(0, 10)

    client.run()
