"""
UDP module for Heimannn HTPA32x32d, multiple sensors (tested up to 3) #TODO
"""
import socket
import os
import sys
import threading # http://www.g-loaded.eu/2016/11/24/how-to-terminate-running-python-threads-using-signals/
import time
import signal

IP_LIST_FP = os.path.join("settings", "devices.txt")
HTPA_PORT = 30444
BUFF_SIZE = 1300

HTPA_CALLING_MSG = "Calling HTPA series devices"
HTPA_BIND_MSG =  "Bind HTPA series device"
HTPA_RELEASE_MSG = "x Release HTPA series device"
HTPA_STREAM_MSG = "K"

HTPA32x32d_PACKET1_LEN = 1292
HTPA32x32d_PACKET2_LEN = 1288
HTPA32x32d_BYTE_FORMAT = "<h" #Little-Endian b

def orderPackets(a, b):
    """
    TODO
    """
    packet1 = a if (len(a) == HTPA32x32d_PACKET1_LEN) else b if (len(b) == HTPA32x32d_PACKET1_LEN) else None
    packet2 = a if (len(a) == HTPA32x32d_PACKET2_LEN) else b if (len(b) == HTPA32x32d_PACKET2_LEN) else None
    return (packet1, packet2)

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

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Credit: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
 
class Device:
    """
    A class for handling HTPA32x32d devices in UDP communication
    """
    def __init__(self, ip):
        self.ip = ip 
        self.port = HTPA_PORT
        self.address = (self.ip, self.port)

class Recorder(threading.Thread):
    def __init__(self, device):
        threading.Thread.__init__(self)
        self.shutdown_flag = threading.Event()
        self.device = device
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1)
        self.sock.bind((socket.gethostbyname(socket.gethostname()), 0))
        # real code starts here

        try:
            self.sock.sendto(HTPA_CALLING_MSG.encode(), self.device.address)
            _ = self.sock.recv(BUFF_SIZE)
            print("Connected successfully to device under %s"%self.device.ip)
        except socket.timeout:
            self.sock.close()
            print("Can't connect to HTPA %s while initializing" % self.device.ip)
            raise ServiceExit
        try: 
            self.sock.sendto(HTPA_BIND_MSG.encode(), self.device.address)
            self.sock.sendto(HTPA_STREAM_MSG.encode(), device.address)   
            print("Streaming HTPA %s" % self.device.ip)  
        except socket.timeout:
            self.sock.close()
            print("Failed to bind HTPA %s while initializing"%self.device.ip)
            raise ServiceExit

    def run(self):
        print('Thread #%s started' % self.ident)
        
        packet1, packet2 = None, None
        while not self.shutdown_flag.is_set():
            try:
                packet_a = self.sock.recv(BUFF_SIZE)
                packet_b = self.sock.recv(BUFF_SIZE)
            except socket.timeout:                    
                self.sock.sendto(HTPA_RELEASE_MSG.encode(), self.device.address)
                print("Terminated HTPA {}".format(self.device.ip))
                self.sock.close()
                print("Timeout when expecting stream from HTPA %s"%self.device.ip)
                raise ServiceExit
            timestamp = global_T0 - time.time()
            old_packet1, old_packet2 = packet1, packet2
            packet1, packet2 = orderPackets(packet_a, packet_b)
            worked = True if (packet1 and packet2) else False
            same = True if ((packet1 == old_packet1) or (packet2 == old_packet2)) else False
            print("Frame from HTPA {} at {} {} same: {}".format(self.device.ip, time.ctime(), worked, same))
            # TODO WRITER!!!
            # TODO CLEAN BOOLS WORKED, SAME, PACKET1, PACKET2 before the loop and OLD_PACKET1, OLD_PACKET2
        
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

def main():
    signal.signal(signal.SIGTERM, service_shutdown)
    signal.signal(signal.SIGINT, service_shutdown)
    print('Starting main program')
    ips = loadIPList()
    if not len(ips):
        sys.exit("Add devices to the file manually, file path: {}".format(IP_LIST_FP))
    for ip in ips:
        if not validateIP(ip):
            sys.exit("IP %s is not a valid IP adress"%ip)

    if len(ips):
        print("Devices listed: ")
        for idx, ip in enumerate(ips):
            print("[%d] %s"%(idx, ip)) 
        proceed = query_yes_no("Proceed with the %d devices listed?"%len(ips), default="yes")
        if not proceed:
            sys.exit("Exiting")
    if proceed:
        devices = []
        for ip in ips:
            devices.append(Device(ip))  
        recorders = [] 
        for device in devices:
            recorders.append(Recorder(device))
        try:
            for recorder in recorders:
                recorder.start()
            while True:
                time.sleep(0.5)
        except ServiceExit:
            for recorder in recorders:
                recorder.shutdown_flag.set()

                
global_T0 = time.time()
if __name__ == "__main__":
    main()

