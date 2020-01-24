"""
UDP module for Heimannn HTPA32x32d, multiple sensors (tested up to 3) #TODO
"""
import socket
import os
import sys

IP_LIST_FP = os.path.join("settings", "devices.txt")

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
        proceed = query_yes_no("Proceed with the %d devices listed?"%len(ips), default="no")
        if not proceed:
            sys.exit("Exiting")

    if proceed:
        pass
        