
import socket
import os
import sys
import threading  # http://www.g-loaded.eu/2016/11/24/how-to-terminate-running-python-threads-using-signals/
import time
import datetime
import signal
from pathlib import Path
import struct
import cv2

IP_LIST_FP = os.path.join("recording", "settings", "devices.txt")
HTPA_PORT = 30444
BUFF_SIZE = 1300

HTPA_CALLING_MSG = "Calling HTPA series devices"
HTPA_BIND_MSG = "Bind HTPA series device"
HTPA_RELEASE_MSG = "x Release HTPA series device"
HTPA_STREAM_MSG = "K"

HTPA32x32d_PACKET1_LEN = 1292
HTPA32x32d_PACKET2_LEN = 1288
HTPA32x32d_BYTE_FORMAT = "<h"  # Little-Endian b


def order_packets(a, b):
    """
    Checks if packets are of different lengths and, if yes, orders a pair of packets received.

    Parameters
    ----------
    a, b : packets (buffers)
        A pair of packets containing one frame captured by HTPA 32x32d.

    Returns
    -------
    tuple 
        A pair of ordered packets containing one frame captured by HTPA 32x32d (packet1, packet2).
    """
    packet1 = a if (len(a) == HTPA32x32d_PACKET1_LEN) else b if (
        len(b) == HTPA32x32d_PACKET1_LEN) else None
    packet2 = a if (len(a) == HTPA32x32d_PACKET2_LEN) else b if (
        len(b) == HTPA32x32d_PACKET2_LEN) else None
    return (packet1, packet2)


def decode_packets(packet1, packet2) -> str:
    """
    Decodes a pair 

    Parameters
    ----------
    packet1, packet2 : packets (buffers)
        A pair of ordered packets containing one frame captured by HTPA 32x32d.

    Returns
    -------
    str 
        Decoded space-delimited temperature values in [1e2 deg. Celsius] (consistent with Heimann's data structure)
    """
    packet = packet1 + packet2
    packet_txt = ""
    for byte in struct.iter_unpack(HTPA32x32d_BYTE_FORMAT, packet):
        packet_txt += str(byte[0]) + " "
    return packet_txt


def loadIPList():
    fp = IP_LIST_FP
    try:
        with open(fp) as file:
            data = file.read()
    except:
        print("The file {} doesn't exist".format(fp))
        print("Creating new settings file...")
        open(fp, 'w').close()
        return []
    return data.splitlines()


def validateIP(ip):
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False


class Device:
    """
    A class for handling HTPA32x32d devices in UDP communication
    """

    def __init__(self, ip):
        self.ip = ip
        self.port = HTPA_PORT
        self.address = (self.ip, self.port)

class Recorder(threading.Thread):
    def __init__(self, device, fp, T0, header=None):
        threading.Thread.__init__(self)
        self.shutdown_flag = threading.Event()
        self.device = device
        self.fp = fp
        self.T0 = T0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1)
        self.sock.bind((socket.gethostbyname(socket.gethostname()), 0))

        try:
            self.sock.sendto(HTPA_CALLING_MSG.encode(), self.device.address)
            _ = self.sock.recv(BUFF_SIZE)
            print("Connected successfully to device under %s" % self.device.ip)
        except socket.timeout:
            self.sock.close()
            print("Can't connect to HTPA %s while initializing" % self.device.ip)
            raise ServiceExit
        try:
            self.sock.sendto(HTPA_BIND_MSG.encode(), self.device.address)
            self.sock.recv(BUFF_SIZE)
            self.sock.sendto(HTPA_STREAM_MSG.encode(), device.address)
            print("Streaming HTPA %s" % self.device.ip)
        except socket.timeout:
            self.sock.close()
            print("Failed to bind HTPA %s while initializing" % self.device.ip)
            raise ServiceExit
        if not header:
            header2write = 'HTPA32x32d\n'
        else:
            header2write = str(header).rstrip('\n')+('\n')
        with open(self.fp, 'w') as file:
            file.write(header2write)

    def run(self):
        print('Thread [TPA] #%s started' % self.ident)

        packet1, packet2 = None, None
        while not self.shutdown_flag.is_set():
            try:
                packet_a = self.sock.recv(BUFF_SIZE)
                packet_b = self.sock.recv(BUFF_SIZE)
            except socket.timeout:
                self.sock.sendto(HTPA_RELEASE_MSG.encode(),
                                 self.device.address)
                print("Terminated HTPA {}".format(self.device.ip))
                self.sock.close()
                print("Timeout when expecting stream from HTPA %s" %
                      self.device.ip)
                raise ServiceExit
            timestamp = time.time() - self.T0
            if not (packet_a and packet_b):
                continue
            packet_str = decode_packets(*order_packets(packet_a, packet_b))
            with open(self.fp, 'a') as file:
                file.write("{}t: {:.2f}\n".format(packet_str, timestamp))

        # CLEANUP !!!
        self.sock.sendto(HTPA_RELEASE_MSG.encode(), self.device.address)
        print("Terminated HTPA {}".format(self.device.ip))


class ServiceExit(Exception):
    """
    Custom exception which is used to trigger the clean exit
    of all running threads and the main program.
    """
    pass


def service_shutdown(signum, frame):
    print('Caught signal %d' % signum)
    raise ServiceExit



class Cap(threading.Thread):
    def __init__(self, device, fp, T0):
        threading.Thread.__init__(self)
        self.shutdown_flag = threading.Event()
        self.T0 = T0
        self.device = device
        fp_prefix, fp_extension = fp.split(".")
        self.fp = fp
        self.fp_prefix = fp_prefix
        self.fp_extension = fp_extension
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1)
        self.sock.bind((socket.gethostbyname(socket.gethostname()), 0))

        try:
            self.sock.sendto(HTPA_CALLING_MSG.encode(), self.device.address)
            _ = self.sock.recv(BUFF_SIZE)
            print("Connected successfully to device under %s" % self.device.ip)
        except socket.timeout:
            self.sock.close()
            print("Can't connect to HTPA %s while initializing" % self.device.ip)
            raise ServiceExit
        try:
            self.sock.sendto(HTPA_BIND_MSG.encode(), self.device.address)
            self.sock.recv(BUFF_SIZE)
            self.sock.sendto(HTPA_STREAM_MSG.encode(), device.address)
            print("Streaming HTPA %s" % self.device.ip)
        except socket.timeout:
            self.sock.close()
            print("Failed to bind HTPA %s while initializing" % self.device.ip)
            raise ServiceExit

    def run(self):
        print('Thread [TPA] #%s started' % self.ident)

        packet1, packet2 = None, None
        photo_idx = 0
        while not self.shutdown_flag.is_set():
            input('Camera {} ready to capture photo. Press ENTER...'.format(self.device.ip))
            try:
                packet_a = self.sock.recv(BUFF_SIZE)
                packet_b = self.sock.recv(BUFF_SIZE)
            except socket.timeout:
                self.sock.sendto(HTPA_RELEASE_MSG.encode(),
                                 self.device.address)
                print("Terminated HTPA {}".format(self.device.ip))
                self.sock.close()
                print("Timeout when expecting stream from HTPA %s" %
                      self.device.ip)
                raise ServiceExit
            timestamp = time.time() - self.T0
            photo_idx += 1
            if not (packet_a and packet_b):
                continue
            packet_str = decode_packets(*order_packets(packet_a, packet_b))
            current_fp = self.fp_prefix + "_{:02d}".format(photo_idx) + "." + self.fp_extension
            with open(current_fp, 'w') as file:
                file.write("HTPA32x32d\n{}t: {:.2f}\n".format(packet_str, timestamp))
            print('{} saved.'.format(current_fp))
            
        self.sock.sendto(HTPA_RELEASE_MSG.encode(), self.device.address)
        print("Terminated HTPA {}".format(self.device.ip))

class WebCam(threading.Thread):
    def __init__(self, dir_path, T0, height = 480 , width = 640, extension = "jpg"):
        threading.Thread.__init__(self)
        self.shutdown_flag = threading.Event()
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        ret, frame = cam.read()
        if not ret:
            print("No webcam found")
            raise ServiceExit
        self.T0 = T0
        self.cam = cam
        self.dir_path = dir_path
        self.ready = True
        self.cap = False
        self.extension = extension

    def run(self):
        print('Thread [camera] #%s started' % self.ident)
        while not self.shutdown_flag.is_set():
                self._write()
                cv2.waitKey(20)
        self.cam.release()

    def _write(self):
        timestamp = time.time() - self.T0
        fp = os.path.join(self.dir_path, "{:.2f}".format(timestamp).replace(".","-") + "." + self.extension)
        ret, frame = self.cam.read()
        if not ret:
            print("Webcam not reachable!")
            raise ServiceExit
        cv2.imwrite(fp, frame)
        cv2.waitKey(1)




