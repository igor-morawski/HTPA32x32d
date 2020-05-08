"""
Python program that connects to Heimann HTPA sensors given their IP addresses (in settings file) and records data captured to TXT files. 
Supports recording mutliple sensors at the same time. This tool is supposed to help developing multi-view thermopile sensor array monitoring system.
"""
import socket
import os
import sys
import threading  # http://www.g-loaded.eu/2016/11/24/how-to-terminate-running-python-threads-using-signals/
import time
import datetime
import signal
from pathlib import Path
import struct
import argparse

import HTPA32x32d.communication
import HTPA32x32d.tools


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", help="Header to write to recorded TXTs, optional",
                    type=str, default=None)
    parser.add_argument("--dest", help="Destination directory (path)",
                    type=str, default=".")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, HTPA32x32d.communication.service_shutdown)
    signal.signal(signal.SIGINT, HTPA32x32d.communication.service_shutdown)
    print('Starting main program (TPA + RGB recorder)')
    ips = HTPA32x32d.communication.loadIPList()
    if not len(ips):
        sys.exit("Add devices to the file manually, file path: {}".format(HTPA32x32d.communication.IP_LIST_FP))
    for ip in ips:
        if not HTPA32x32d.communication.validateIP(ip):
            sys.exit("IP %s is not a valid IP adress" % ip)

    if len(ips):
        print("Devices listed: ")
        for idx, ip in enumerate(ips):
            print("[%d] %s" % (idx, ip))
        proceed = query_yes_no(
            "Proceed with the %d devices listed?" % len(ips), default="yes")
        if not proceed:
            sys.exit("Exiting")

    if proceed:
        devices = []
        for ip in ips:
            devices.append(HTPA32x32d.communication.Device(ip))
        # dir and fn
        directory_path = os.path.join(args.dest, global_T0_YYYYMMDD)
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        rgb_path = os.path.join(directory_path,"{}_ID{}".format(global_T0_YYYYMMDD_HHMM, "RGB"))
        Path(rgb_path).mkdir(parents=True, exist_ok=True)
        webcam = HTPA32x32d.communication.WebCam(rgb_path, global_T0, extension=HTPA32x32d.tools.HTPA_UDP_MODULE_WEBCAM_IMG_EXT)
        recorders = []
        for device in devices:
            fn = "{}_ID{}.TXT".format(
                global_T0_YYYYMMDD_HHMM, device.ip.split(".")[-1])
            fp = os.path.join(directory_path, fn)
            recorders.append(HTPA32x32d.communication.Recorder(device, fp, global_T0, header=args.header))
        try:
            webcam.start()
            for recorder in recorders:
                recorder.start()
            while True:
                time.sleep(0.5)
        except HTPA32x32d.communication.ServiceExit:
            webcam.shutdown_flag.set()
            for recorder in recorders:
                recorder.shutdown_flag.set()


global_T0 = time.time()
global_T0_strct = time.strptime(time.ctime(global_T0))
global_T0_HHMM = "{:02d}{:02d}".format(
    global_T0_strct.tm_year, global_T0_strct.tm_hour, global_T0_strct.tm_min)
global_T0_YYYYMMDD = "{:04d}{:02d}{:02d}".format(
    global_T0_strct.tm_year, global_T0_strct.tm_mon, global_T0_strct.tm_mday)
global_T0_YYYYMMDD_HHMM = "{:04d}{:02d}{:02d}_{:02d}{:02d}".format(global_T0_strct.tm_year,
                                                                   global_T0_strct.tm_mon, global_T0_strct.tm_mday,
                                                                   global_T0_strct.tm_hour,
                                                                   global_T0_strct.tm_min)
if __name__ == "__main__":
    main()
