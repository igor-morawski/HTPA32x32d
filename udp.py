"""
UDP module for Heimannn HTPA32x32d, multiple sensors (tested up to 3) #TODO
"""
import socket
import os
import sys

IP_LIST_FP = os.path.join("settings", "devices.txt")
HTPA_PORT = 30444

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
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            msg = "Calling HTPA series devices"
            try:
                sock.bind((socket.gethostbyname(socket.gethostname()), 0))
                sock.connect(device.address)
                sock.sendto(msg.encode(), device.address)
                response = sock.recv(1024)
                print(response)
                response = sock.recv(1024)
                print(response)
            finally:
                sock.close()