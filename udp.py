"""
UDP module for Heimannn HTPA32x32d, multiple sensors (tested up to 3) #TODO
"""
import socket
import os
import sys

IP_LIST_FP = os.path.join("settings", "devices.txt")
HTPA_PORT = 30444
BUFF_SIZE = 1300

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

if __name__ == "__main__":
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
        for device in devices:
            # TODO: change to multithread
            # INIT
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(1)
                sock.bind((socket.gethostbyname(socket.gethostname()), 0))

                msg = "Calling HTPA series devices"
                sock.sendto(msg.encode(), device.address)
                response = sock.recv(BUFF_SIZE)
                print("Connected successfully to device under %s"%device.ip)
            except socket.timeout:
                sock.close()
                sys.exit("Can't connect to HTPA %s while initializing" % device.ip)
            finally:
                sock.close()
            # BIND
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(1)
                sock.bind((socket.gethostbyname(socket.gethostname()), 0))

                msg = "Bind HTPA series device"
                sock.sendto(msg.encode(), device.address)
                response = sock.recv(BUFF_SIZE)
                msg = "K"
                sock.sendto(msg.encode(), device.address)      
                #TODO delete a b  
                a = sock.recv(BUFF_SIZE)
                b = sock.recv(BUFF_SIZE)
                with open("a.txt", 'wb+') as file:
                    file.write(a)
                with open("b.txt", 'wb+') as file:
                    file.write(b)
                #TODO cont. here
            except socket.timeout:
                msg = "x Release HTPA series device"
                sock.sendto(msg.encode(), device.address)
                sock.close()
                print("Device %s released"%device.ip)
                sys.exit("Can't connect to while binding and sending 'k' to HTPA %s" % device.ip)
            finally:
                msg = "x Release HTPA series device"
                sock.sendto(msg.encode(), device.address)
                print("Device %s released"%device.ip)
                sock.close()