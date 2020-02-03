"""
Tools for Heimann HTPA recordings. 

Data types:
    * txt - Heimann HTPA recordings in *.TXT,
    * np - NumPy array of thermopile sensor array data, shaped [frames, height, width],
    * csv - thermopile sensor array data in *.csv format, NOT a pandas dataframe!
    * df - pandas dataframe,
    * pc - NumPy array of pseudocolored thermopile sensor array data, shaped [frames, height, width, channels],
"""
import numpy as np
import pandas as pd
import cv2
import os
import imageio

DTYPE = "float32"
PD_SEP = ","
PD_NAN = np.inf
PD_DTYPE = np.float32
READ_CSV_ARGS = {"skiprows": 1}
PD_TIME_COL = "Time (sec)"
PD_PTAT_COL = "PTAT"

# TODO: np -> array (naming convention)


def txt2np(filepath: str, array_size: int = 32):
    """
    Convert Heimann HTPA .txt to NumPy array shaped [frames, height, width].

    Parameters
    ----------
    filepath : str
    array_size : int, optional

    Returns
    -------
    np.array
        3D array of temperature distribution sequence, shaped [frames, height, width].
    """
    with open(filepath) as f:
        # discard the first line
        _ = f.readline()
        # read line by line now
        line = "dummy line"
        frames = []
        timestamps = []
        while line:
            line = f.readline()
            if line:
                split = line.split(" ")
                frame = split[0 : array_size ** 2]
                timestamp = split[-1]
                frame = np.array([int(T) for T in frame], dtype=DTYPE)
                frame = frame.reshape([array_size, array_size], order="F")
                frame *= 1e-2
                frames.append(frame)
                timestamps.append(float(timestamp))
        frames = np.array(frames)
        # the array needs rotating 90 CW
        frames = np.rot90(frames, k=-1, axes=(1, 2))
    return frames, timestamps


def write_np2csv(output_fp: str, array, timestamps: list) -> bool:
    """
    Convert and save Heimann HTPA NumPy array shaped [frames, height, width] to .CSV dataframe.
    CSV should preferably represent the data collected without preprocessing, cropping or any data manipulation.  

    Parameters
    ----------
    output_fp : str
        Filepath to destination file, including the file name.
    array : np.array
        Temperatue distribution sequence, shaped [frames, height, width].
    timestamps : list
        List of timestamps of corresponding array frames.
    """
    # initialize csv template (and append frames later)
    # prepend first row for compability with legacy format
    first_row = pd.DataFrame({"HTPA 32x32d": []})
    first_row.to_csv(output_fp, index=False, sep=PD_SEP)
    headers = {PD_TIME_COL: [], PD_PTAT_COL: []}
    df = pd.DataFrame(headers)
    for idx in range(np.prod(array.shape[1:])):
        df.insert(len(df.columns), "P%04d" % idx, [])
    df.to_csv(output_fp, mode="a", index=False, sep=PD_SEP)

    for idx in range(array.shape[0]):
        frame = array[idx, ...]
        timestamp = timestamps[idx]
        temps = list(frame.flatten())
        row_data = [timestamp, PD_NAN]
        row_data.extend(temps)
        row = pd.DataFrame([row_data])
        row = row.astype(PD_DTYPE)
        row.to_csv(output_fp, mode="a", header=False, sep=PD_SEP, index=False)
    return True


def read_csv2np(csv_fp: str):
    """
    Read and convert .CSV dataframe to a Heimann HTPA NumPy array shaped [frames, height, width]

    Parameters
    ----------
    csv_fp : str
        Filepath to the csv file tor read.

    Returns
    -------
    array : np.array
        Temperatue distribution sequence, shape [frames, height, width].
    timestamps : list
        List of timestamps of corresponding array frames.
    """
    df = pd.read_csv(csv_fp, **READ_CSV_ARGS)
    timestamps = df[PD_TIME_COL]
    array = df.drop([PD_TIME_COL, PD_PTAT_COL], axis=1).to_numpy(dtype=DTYPE)
    array = reshape_flattened_frames(array)
    return array, timestamps


def apply_heatmap(array, cv_colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Applies pseudocoloring (heatmap) to a sequence of thermal distribution. Same as np2pc().
    np2pc() is preffered.

    Parameters
    ----------
    array : np.array
         (frames, height, width)
    cv_colormap : int, optional

    Returns
    -------
    np.array
         (frames, height, width, channels)
    """
    min, max = array.min(), array.max()
    shape = array.shape
    array_normalized = (255 * ((array - min) / (max - min))).astype(np.uint8)
    heatmap_flat = cv2.applyColorMap(array_normalized.flatten(), cv_colormap)
    return heatmap_flat.reshape([shape[0], shape[1], shape[2], 3])


def np2pc(array, cv_colormap: int = cv2.COLORMAP_JET) -> np.ndarray:

    """
    Applies pseudocoloring (heatmap) to a sequence of thermal distribution. Same as apply_heatmap().
    np2pc() is preffered.

    Parameters
    ----------
    array : np.array
         (frames, height, width)
    cv_colormap : int, optional

    Returns
    -------
    np.array
         (frames, height, width, channels)
    """
    return apply_heatmap(array, cv_colormap)


# TODO heatmap -> pc


def save_frames(array, dir_name: str, extension: str = ".bmp") -> bool:
    """
    Exctracts and saves frames from a sequence array into a folder dir_name

    Parameters
    ----------
    array : np.array
         (frames, height, width, channels)

    Returns
    -------
    bool
        True if success
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    for idx, frame in enumerate(array):
        cv2.imwrite(os.path.join(dir_name, "%d" % idx + extension), frame)
    return True


def flatten_frames(array):
    """
    Flattens array of shape [frames, height, width] into array of shape [frames, height*width]

    Parameters
    ----------
    array : np.array
         (frames, height, width)

    Returns
    -------
    np.array
        flattened array (frames, height, width)
    """
    _, height, width = array.shape
    return array.reshape((-1, height * width))


def write_pc2gif(array, fp: str, fps=10, loop: int = 0, duration=None):
    """
    Converts and saves NumPy array of pseudocolored thermopile sensor array data, shaped [frames, height, width, channels], into a .gif file

    Parameters
    ----------
    array : np.array
        Pseudocolored data (frames, height, width, channels).
    fp : str
        The filepath to write to.
    fps : float, optional
        Default 10, approx. equal to a typical thermopile sensor array FPS value.
    loop : int, optional
        The number of iterations. Default 0 (meaning loop indefinitely).
    duration : float, list, optional
        The duration (in seconds) of each frame. Either specify one value
        that is used for all frames, or one value for each frame.
        Note that in the GIF format the duration/delay is expressed in
        hundredths of a second, which limits the precision of the duration. (from imageio doc)

    Returns
    -------
    bool
        True if success.
    """
    if not duration:
        duration = 1 / fps
    with imageio.get_writer(fp, mode="I", duration=duration, loop=loop) as writer:
        for frame in array:
            writer.append_data(frame[:, :, ::-1])
    return True


def timestamps2frame_durations(timestamps: list, last_frame_duration=None) -> list:
    """
    Produces frame durations list to make gifs produced with write_pc2gif() more accurate temporally, 
    # TODO 
    Parameters
    ----------
    timestamps : list
        List of timestamps of corresponding array frames.
    last_frame_duration : float, optional
        List of N timestamps gives information about durations of N-1 initial frames, 
        if not given, the function will duplicate the last value in the produced list to make up for the missing frame duration.

    Returns
    -------
    list
        List of frame durations.
    """
    frame_durations = [x_t2 - x_t1 for x_t1, x_t2 in zip(timestamps, timestamps[1:])]
    if not last_frame_duration:
        last_frame_duration = frame_durations[-1]
    frame_durations.append(last_frame_duration)
    return frame_durations

def reshape_flattened_frames(array):
    """
    Reshapes array shaped [frames, height*width] into array of shape [frames, height, width]

    Parameters
    ----------
    array : np.array
         flattened array (frames, height*width)

    Returns
    -------
    np.array
        reshaped array (frames, height, width)
    """
    _, elements = array.shape
    height = int(elements ** (1 / 2))
    width = height
    return array.reshape((-1, height, width))

def crop_center(array, crop_height, crop_width):
    """
    Crops the center portion of an infrared sensor array image sequence.


    Parameters
    ---------
    array : np.array
        (frames, height, width)
    crop_height : int
        height of the cropped patch, if -1 then equal to input's height
    crop_width : int
        width of the cropped patch, if -1 then equal to input's width

    Returns
    -------
    np.array
        cropped array (frames, crop_height, crop_width)
    """
    _, height, width = array.shape
    crop_height = height if (crop_height == -1) else crop_height
    start_y = height//2 - crop_height//2
    crop_width = width if (crop_width == -1) else crop_width
    start_x = width//2 - crop_width//2
    return array[:, start_y:start_y+crop_height, start_x:start_x+crop_width]


if __name__ == "__main__":
    import argparse
    import glob

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

        array, timestamps = txt2np(txt_fp)
        # NO CROPPING HERE! CSV SHOULD NOT BE CROPPED!
        if csv:
            csv_fp = init(txt_fp, ".csv")
            if csv_fp:
                write_np2csv(csv_fp, array, timestamps)

        cropped_array = crop_center(array, crop, crop)
        if gif:
            gif_fp = init(txt_fp, ".gif")
            if gif_fp:
                pc = np2pc(cropped_array)
                write_pc2gif(pc, gif_fp, duration=timestamps2frame_durations(timestamps))
        if bmp:
            parent, txt_fn = os.path.split(txt_fp)
            fn = txt_fn.split(".TXT")[0]
            dest = fn
            if init(txt_fp, ""):
                print("Extracting frames...")
                save_frames(np2pc(cropped_array), os.path.join(parent, dest))
        return

    if dir_path:
        for txt_fp in glob.glob(os.path.join(dir_path, "*.TXT")):
            txtFunctions(txt_fp, **vars(args))
    if file_path:
        txt_fp = file_path
        txtFunctions(txt_fp, **vars(args))

