# THIS IS A RESOURCE FILE TO SHOW HOW TO HANDLE THE INCOMING PACKETS
import os
import struct
import time

PACKET1_LEN = 1292
PACKET2_LEN = 1288
HTPA_BYTE_FORMAT = "<h" #Little-Endian b

def orderPackets(a, b):
    """
    TODO
    """
    packet1 = a if (len(a) == PACKET1_LEN) else b if (len(b) == PACKET1_LEN) else None
    packet2 = a if (len(a) == PACKET2_LEN) else b if (len(b) == PACKET2_LEN) else None
    return (packet1, packet2)

def decodePackets(packet1, packet2):
    """
    TODO
    """
    packet = packet1 + packet2
    packet_txt = ""
    for byte in struct.iter_unpack(HTPA_BYTE_FORMAT, packet):
        packet_txt += str(byte[0]) + " "
    return packet_txt

def addTimestamp(decodedPackets, time):
    return decodedPackets + "t: " + str("{0:.2f}".format(time))
    #TODO CHANGE THE DUMMY TIMESTAMP!!!!

if __name__ == "__main__":
    t0 = time.time()
    with open('a.txt', 'rb') as file:
        a=file.read()
    with open('b.txt', 'rb') as file:
        b=file.read()
    packet1, packet2 = orderPackets(a, b)
    frame_binary = packet1 + packet2
    # struct.unpack(">b", po jednym)
    # x = struct.iter_unpack(">b", a[:2])
    result = decodePackets(*orderPackets(a, b))
    with open('result.TXT', 'w+') as file:
        file.write("ARRAYTYPE=10MBIT=12REFCAL=2T=Y"+'\n')
        file.write(addTimestamp(result, t0 - time.time())+'\n')
    with open('result.TXT', 'r') as file:
        x = file.readline()
        y = file.readline()
        c = file.readline()


    