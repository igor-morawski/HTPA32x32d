'''
Converts all TXT files in a given directory or a given file.
Call python converter.py --help to learn more.
'''
import argparse
import glob
import os
import tools

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process .TXT files in a directory passed"
    )
    parser.add_argument("object")
    parser.add_argument(
        "--gif", "-g", dest="gif", help="Write gifs", action="store_true"
    )
    parser.add_argument(
        "--csv", "-c", dest="csv", help="Write csvs", action="store_true"
    )
    parser.add_argument(
        "--bmp",
        "-b",
        dest="bmp",
        help="Write bitmaps to a directory named same as the processed file",
        action="store_true",
    )
    parser.add_argument(
        "--crop",
        dest="crop",
        help="Crop a rectangular patch of size (crop, crop) from the center of an array",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        help="Overwrite file if it exists",
        action="store_true",
    )
    args = parser.parse_args()
    dir_path, file_path = None, None
    if os.path.isdir(args.object):
        dir_path = os.path.abspath(args.object)
    elif os.path.isfile(args.object):
        file_path = os.path.abspath(args.object)

    def txtFunctions(txt_fp, gif=False, csv=False, bmp=False, crop=-1, overwrite=False, **args):
        def init(txt_fp, ext):
            parent, txt_fn = os.path.split(txt_fp)
            fn = txt_fn.split(".TXT")[0]
            txt_fn = fn + ".TXT"
            ext_fn = fn + ext
            ext_fp = os.path.join(parent, ext_fn)
            if os.path.exists(ext_fp):
                if not overwrite:
                    print("{} already exists, aborting conversion".format(ext_fn))
                    return None
            print("Converting {} to {}".format(txt_fn, ext_fn))
            return ext_fp

        array, timestamps = tools.txt2np(txt_fp)
        # NO CROPPING HERE! CSV SHOULD NOT BE CROPPED!
        if csv:
            csv_fp = init(txt_fp, ".csv")
            if csv_fp:
                tools.write_np2csv(csv_fp, array, timestamps)

        cropped_array = tools.crop_center(array, crop, crop)
        if gif:
            gif_fp = init(txt_fp, ".gif")
            if gif_fp:
                pc = tools.np2pc(cropped_array)
                tools.write_pc2gif(
                    pc, gif_fp, duration=tools.timestamps2frame_durations(timestamps))
        if bmp:
            parent, txt_fn = os.path.split(txt_fp)
            fn = txt_fn.split(".TXT")[0]
            dest = fn
            if init(txt_fp, ""):
                print("Extracting frames...")
                tools.save_frames(tools.np2pc(cropped_array),
                                  os.path.join(parent, dest))
        return

    if dir_path:
        for txt_fp in glob.glob(os.path.join(dir_path, "*.TXT")):
            txtFunctions(txt_fp, **vars(args))
    if file_path:
        txt_fp = file_path
        txtFunctions(txt_fp, **vars(args))
