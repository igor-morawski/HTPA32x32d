import HTPA32x32d
import argparse
import os
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modify .TXT files in a directory passed (subject is modified)"
    )
    parser.add_argument("object")
    parser.add_argument(
        "--new_subject",
        dest="subj",
        type=str
    )
    args = parser.parse_args()
    subj = args.subj
    if os.path.isdir(args.object):
        dir_path = os.path.abspath(args.object)
    else:
        raise ValueError
    txts = glob.glob(os.path.join(dir_path, "*.TXT"))
    for txt in txts:
        header = HTPA32x32d.tools.read_txt_header(txt)
        chunks = header.split(",")
        chunks[0] = subj
        new_header = ','.join([str(elem) for elem in chunks]) 
        HTPA32x32d.tools.modify_txt_header(txt, new_header)