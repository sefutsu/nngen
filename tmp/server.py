import socket
import sys
import numpy as np
from pynq import Overlay, allocate
import nngen_ctrl as ng
import time

encoding = "utf-8"

class BlockingServerBase:
    def __init__(self, timeout:int=60, buffer:int=1024):
        self._socket = None
        self._timeout = timeout
        self._buffer = buffer
        self.conn = None
        self.close()

    def __del__(self):
        self.close()

    def close(self) -> None:
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
            self._socket.close()
        except:
            pass

    def accept(self, address, family:int, typ:int, proto:int) -> None:
        self._socket = socket.socket(family, typ, proto)
        self._socket.settimeout(self._timeout)
        self._socket.bind(address)
        self._socket.listen(1)
        print("Server started:", address)
        self.conn, ret_addr = self._socket.accept()
        print("Accepted:", ret_addr)
        self.communicate()
    
    def communicate(self):
        while True:
            try:
                message_recv = self.conn.recv(self._buffer).decode(encoding)
                print(message_recv)
                message_resp = "self.respond(message_recv)"
                self.conn.send(message_resp.encode(encoding))
            except ConnectionResetError:
                break
            except BrokenPipeError:
                break
        self.close()

    def respond(self, message:str) -> str:
        return ""

class PYNQServer(BlockingServerBase):
    def __init__(self, host:str="0.0.0.0", port:int=8010) -> None:
        self.server=(host,port)
        super().__init__(timeout=60, buffer=1024)
        self.init_fpga()
        self.accept(self.server, socket.AF_INET, socket.SOCK_STREAM, 0)

    def init_fpga(self):
        bitfile = 'vgg11.bit'
        ipname = 'vgg11_0'
        overlay = Overlay(bitfile)
        self.ip = ng.nngen_core(overlay, ipname)
        memory_size = 400 * 1024 * 1024
        self.global_buffer = allocate(shape=(memory_size,), dtype=np.uint8)
        self.ip.set_global_buffer(self.global_buffer)

    def communicate(self):
        while True:
            try:
                command = self.conn.recv(self._buffer).decode(encoding)
                msg = None
                if command == "run":
                    msg = self.run()
                elif command[:6] == "setbuf":
                    _, idx_begin, idx_end = command.split()
                    msg = self.set_buffer(int(idx_begin), int(idx_end))
                elif command[:7] == "readbuf":
                    _, idx_begin, idx_end = command.split()
                    self.read_buffer(int(idx_begin), int(idx_end))
                else:
                    msg = f"unknown command: {command}"
                if msg:
                    self.conn.send(msg.encode(encoding))
            except ConnectionResetError:
                break
            except BrokenPipeError:
                break
        self.close()

    def run(self):
        start_time = time.time()
        self.ip.run()
        self.ip.wait()
        end_time = time.time()
        elapsed_time = end_time - start_time
        msg = f"run in {elapsed_time} secs."
        return msg

    def set_buffer(self, idx_begin, idx_end):
        idx = idx_begin
        while idx < idx_end:
            values = self.conn.recv(self._buffer)
            l = len(values)
            self.global_buffer[idx : idx + l] = np.frombuffer(values, dtype=np.uint8)
        msg = f"write {idx_end - idx_begin} bytes to global buffer"
        return msg

    def read_buffer(self, idx_begin, idx_end):
        print(self.global_buffer[idx_begin:idx_end])


if __name__ == "__main__":
    port = 8010
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    PYNQServer(port=port)
