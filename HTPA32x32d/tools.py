"""
A collection of useful functions and data structures for working with data captured by Heimann HTPA32x32d and other thermopile sensor arrays.

Data types:
    * txt - Heimann HTPA recordings in *.TXT,
    * np - NumPy array of thermopile sensor array data, shaped [frames, height, width],
    * csv - thermopile sensor array data in *.csv file, NOT a pandas dataframe!
    * df - pandas dataframe,
    * pc - NumPy array of pseudocolored thermopile sensor array data, shaped [frames, height, width, channels],

Warnings:
    when converting TXT -> other types the array is rotated 90 deg. CW
    numpy array order is 'K'
    txt array order is 'F'
"""
import numpy as np
import pandas as pd
import cv2
import os
import imageio
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pickle
import itertools
import json
import glob
import collections
import shutil
import pickle
import re


DTYPE = "float32"
PD_SEP = ","
PD_NAN = np.inf
PD_DTYPE = np.float32
READ_CSV_ARGS = {"skiprows": 1}
PD_TIME_COL = "Time (sec)"
PD_PTAT_COL = "PTAT"


HTPA_UDP_MODULE_WEBCAM_IMG_EXT = "jpg"


READERS_EXTENSIONS_DICT = {
    "txt": "txt",
    "csv": "csv",
    "pickle": "pickle",
    "pkl": "pickle",
    "p": "pickle",
}


SUPPORTED_EXTENSIONS = list(READERS_EXTENSIONS_DICT.keys())


def remove_extension(filepath):
    return filepath.split(".")[0]


def get_extension(filepath):
    return filepath.split(".")[1]


def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_parent_exists(path):
    ensure_path_exists(os.path.dirname(path))


def read_tpa_file(filepath: str, array_size: int = 32):
    """
    Convert Heimann HTPA file to NumPy array shaped [frames, height, width].
    Currently supported: see SUPPORTED_EXTENSIONS flag

    Parameters
    ----------
    filepath : str
    array_size : int, optional (for txt files only)

    Returns
    -------
    np.array
        3D array of temperature distribution sequence, shaped [frames, height, width].
    list
        list of timestamps
    """
    extension_lowercase = get_extension(filepath).lower()
    assert (extension_lowercase in SUPPORTED_EXTENSIONS)
    reader = READERS_EXTENSIONS_DICT[extension_lowercase]
    if reader == 'txt':
        return txt2np(filepath)
    if reader == 'csv':
        return csv2np(filepath)
    if reader == 'pickle':
        return pickle2np(filepath)


def write_tpa_file(filepath: str, array, timestamps: list, header=None) -> bool:
    """
    Convert and save Heimann HTPA NumPy array shaped [frames, height, width] to a txt file.
    Currently supported: see SUPPORTED_EXTENSIONS flag

    Parameters
    ----------
    filepath : str
        Filepath to destination file, including the file name.
    array : np.array
        Temperatue distribution sequence, shaped [frames, height, width].
    timestamps : list
        List of timestamps of corresponding array frames.
    """
    extension_lowercase = get_extension(filepath).lower()
    assert (extension_lowercase in SUPPORTED_EXTENSIONS)
    writer = READERS_EXTENSIONS_DICT[extension_lowercase]
    if writer == 'txt':
        return write_np2txt(filepath, array, timestamps, header=header)
    if writer == 'csv':
        assert not header
        return write_np2csv(filepath, array, timestamps)
    if writer == 'pickle':
        assert not header
        return write_np2pickle(filepath, array, timestamps)

def modify_txt_header(filepath : str, new_header):
    header = new_header.rstrip()
    header += "\n"
    with open(filepath) as f:
        lines = f.readlines()
        lines[0] = header
    with open(filepath, "w") as f:
        f.writelines(lines)

def read_txt_header(filepath: str):
    """
    Read Heimann HTPA .txt header.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    str
        TPA file header
    """
    with open(filepath) as f:
        header = f.readline().rstrip()
    return header


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
    list
        list of timestamps
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
                frame = split[0: array_size ** 2]
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


def write_np2txt(output_fp: str, array, timestamps: list, header: str = None) -> bool:
    """
        Convert and save Heimann HTPA NumPy array shaped [frames, height, width] to a txt file.

        Parameters
        ----------
        output_fp : str
            Filepath to destination file, including the file name.
        array : np.array
            Temperatue distribution sequence, shaped [frames, height, width].
        timestamps : list
            List of timestamps of corresponding array frames.
        header : str, optional
            TXT header
        """
    ensure_parent_exists(output_fp)
    frames = np.rot90(array, k=1, axes=(1, 2))
    if header:
        header = header.rstrip()
        header += "\n"
    else:
        header = "HTPA32x32d\n"
    with open(output_fp, 'w') as file:
        file.write(header)
        for step, t in zip(frames, timestamps):
            line = ""
            for val in step.flatten("F"):
                line += ("%02.2f" % val).replace(".", "")[:4] + " "
            file.write("{}t: {}\n".format(line, t))


def write_np2pickle(output_fp: str, array, timestamps: list) -> bool:
    """
    Convert and save Heimann HTPA NumPy array shaped [frames, height, width] to a pickle file.

    Parameters
    ----------
    output_fp : str
        Filepath to destination file, including the file name.
    array : np.array
        Temperatue distribution sequence, shaped [frames, height, width].
    timestamps : list
        List of timestamps of corresponding array frames.
    """
    ensure_parent_exists(output_fp)
    with open(output_fp, "wb") as f:
        pickle.dump((array, timestamps), f)
    return True


def pickle2np(filepath: str):
    """
    Convert Heimann HTPA .txt to NumPy array shaped [frames, height, width].

    Parameters
    ----------
    filepath : str

    Returns
    -------
    np.array
        3D array of temperature distribution sequence, shaped [frames, height, width].
    list
        list of timestamps
    """
    with open(filepath, "rb") as f:
        frames, timestamps = pickle.load(f)
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
    ensure_parent_exists(output_fp)
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


def csv2np(csv_fp: str):
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
    ensure_parent_exists(fp)
    if not duration:
        duration = 1 / fps
    with imageio.get_writer(fp, mode="I", duration=duration, loop=loop) as writer:
        for frame in array:
            writer.append_data(frame[:, :, ::-1])
    return True


def timestamps2frame_durations(timestamps: list, last_frame_duration=None) -> list:
    """
    Produces frame durations list to make gifs produced with write_pc2gif() more accurate temporally, 
    
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
    frame_durations = [x_t2 - x_t1 for x_t1,
                       x_t2 in zip(timestamps, timestamps[1:])]
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


def crop_center(array, crop_height=None, crop_width=None):
    """
    Crops the center portion of an infrared sensor array image sequence.


    Parameters
    ---------
    array : np.array
        (frames, height, width) or (frames, height, width, channel)
    crop_height : int, optional
        Height of the cropped patch, if -1 then equal to input's height.
        If crop_height, crop_width are None image will be cropped to match smaller spatial dimension.
    crop_width : int, optional
        Width of the cropped patch, if -1 then equal to input's width.
        If crop_height, crop_width are None image will be cropped to match smaller spatial dimension.

    Returns
    -------
    np.array
        cropped array (frames, crop_height, crop_width)
    """
    _, height, width = array.shape[:3]
    if not (crop_width or crop_height):
        smaller_dim = height if (height < width) else width
        crop_width, crop_height =  smaller_dim, smaller_dim
    if not crop_width:
        if crop_height:
            crop_width = crop_height
    if not crop_height:
        if crop_width:
            crop_height = crop_width
    crop_height = height if (crop_height == -1) else crop_height
    start_y = height//2 - crop_height//2
    crop_width = width if (crop_width == -1) else crop_width
    start_x = width//2 - crop_width//2
    return array[:, start_y:start_y+crop_height, start_x:start_x+crop_width]


def match_timesteps(*timestamps_lists):
    """
    Aligns timesteps of given timestamps.


    Parameters
    ---------
    *timestamps_list : list, np.array
        lists-like data containing timestamps 
    Returns
    -------
    list
        list of indices of timesteps corresponding to input lists so that input lists are aligned
    
    Example:
        ts1 = [1, 2, 3, 4, 5]
        ts2 = [1.1, 2.1, 2.9, 3.6, 5.1, 6, 6.1]
        ts3 = [0.9, 1.2, 2, 3, 4.1, 4.2, 4.3, 4.9]
        idx1, idx2, idx3 = match_timesteps(ts1, ts2, ts3)
    now ts1[idx1], ts2[idx2] and ts3[idx3] will be aligned
    """
    ts_list = [np.array(ts).reshape(-1, 1) for ts in timestamps_lists]
    min_len_idx = np.array([len(ts) for ts in ts_list]).argmin()
    min_len_ts = ts_list[min_len_idx]
    indices_list = [None] * len(ts_list)
    for idx, ts in enumerate(ts_list):
        if (idx == min_len_idx):
            indices_list[idx] = list(range(len(min_len_ts)))
        else:
            indices_list[idx] = list(cdist(min_len_ts, ts).argmin(axis=-1))
    return indices_list


def match_timesteps2(*timestamps_lists):
    #XXX Not finished
    """
    Aligns timesteps of given timestamps. 


    Parameters
    ---------
    *timestamps_list : list, np.array
        lists-like data containing timestamps 
    Returns
    -------
    list
        list of indices of timesteps corresponding to input lists so that input lists are aligned
    
    Example:
        ts1 = [1, 2, 3, 4, 5]
        ts2 = [1.1, 2.1, 2.9, 3.6, 5.1, 6, 6.1]
        ts3 = [0.9, 1.2, 2, 3, 4.1, 4.2, 4.3, 4.9]
        idx1, idx2, idx3 = match_timesteps(ts1, ts2, ts3)
    now ts1[idx1], ts2[idx2] and ts3[idx3] will be aligned
    """
    ts_list = [np.array(ts).reshape(-1, 1) for ts in timestamps_lists]
    #min_len_idx = np.array([len(ts) for ts in ts_list]).argmin()
    #min_len_ts = ts_list[min_len_idx]
    max_error_list = [0] * len(ts_list)
    for idx, ts in enumerate(ts_list):
        for idx2, ts2 in enumerate(ts_list):
            if (idx == idx2):
                continue
            tmp_indexes = list(cdist(ts, ts2).argmin(axis=-1))
            diff = ts - ts2[tmp_indexes]
            max_error = np.abs(np.max(diff))
            current_max = max_error_list[idx]
            if (max_error > current_max):
                max_error_list[idx] = max_error
    min_error_idx = np.argmin(max_error_list)
    indices_list = [None] * len(ts_list)
    min_error_ts = ts_list[min_error_idx]
    for idx, ts in enumerate(ts_list):
        if (idx == min_error_idx):
            indices_list[idx] = list(range(len(min_error_ts)))
        else:
            indices_list[idx] = list(cdist(min_error_ts, ts).argmin(axis=-1))
    return indices_list


def resample_np_tuples(arrays, indices=None, step=None):
    """
    Resampling for 3D arrays.

    Parameters
    ---------
    arrays : list
        arays to resample 
    indices : list, optional
        list of indices applied to arrays
    step : int, optional
        resampling with a step, if given indices will be ignored
    Returns
    -------
    list
        list of resampled arrays
    """
    if indices:
        if len(arrays) != len(indices):
            raise ValueError('Iterables have different lengths')
        resampled_arrays = []
        for array, ids in zip(arrays, indices):
            resampled_arrays.append(array[ids])
        return resampled_arrays
    if step:
        return [array[range(0, len(array), step)] for array in arrays]
    return arrays


def save_temperature_histogram(array, fp="histogram.png", bins=None, xlabel='Temperature grad. C', ylabel='Number of pixels', title='Histogram of temperature', grid=True, mu=False, sigma=False):
    """
    Saves a histogram of measured temperatures


    Parameters
    ---------
    array : np.array
        (frames, height, width)
    fp : str
        filepath to save plotted histogram to
    bins, xlabel, ylabel, title, grid
        as in pyplot
    """
    data = array.flatten()
    hist = plt.hist(data, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    text = r'{}{}{}'.format('$\mu={0:.2f} \degree C$'.format(data.mean()) if mu else '', ', ' if (
        mu and sigma) else '', '$\sigma={0:.2f} \degree C$'.format(data.std()) if sigma else '')
    plt.title("{} {}".format(title, text))
    plt.grid(grid)
    plt.savefig(fp)
    plt.close('all')
    return True


def resample_timestamps(timestamps, indices=None, step=None):
    """
    Resampling for 3D arrays.

    Parameters
    ---------
    arrays : list
        arays to resample 
    indices : list, optional
        list of indices applied to arrays
    step : int, optional
        resampling with a step, if given indices will be ignored
    Returns
    -------
    list
        list of resampled arrays
    """
    ts_array = [np.array(ts) for ts in timestamps]
    return [list(ts) for ts in resample_np_tuples(ts_array, indices, step)]


def debug_HTPA32x32d_txt(filepath: str, array_size=32):
    """
    Debug Heimann HTPA .txt by attempting to convert to NumPy array shaped [frames, height, width].

    Parameters
    ----------
    filepath : str
    array_size : int, optional

    Returns
    -------
    int
        line that raises error, -1 if no error
    """
    with open(filepath) as f:
        line_n = 1
        _ = f.readline()
        line = "dummy line"
        frames = []
        timestamps = []
        while line:
            line_n += 1
            line = f.readline()
            if line:
                try:
                    split = line.split(" ")
                    frame = split[0: array_size ** 2]
                    timestamp = split[-1]
                    frame = np.array([int(T) for T in frame], dtype=DTYPE)
                    frame = frame.reshape([array_size, array_size], order="F")
                    frame *= 1e-2
                    frames.append(frame)
                    timestamps.append(float(timestamp))
                except:
                    split = line.split(" ")
                    frame = split[0: array_size ** 2]
                    timestamp = split[-1]
                    T_idx = 0
                    for T in frame:
                        try:
                            _ = int(T)
                        except:
                            break
                        T_idx += 1
                    print("{} caused error at line {} (t: {}), bit {} (= {})".format(
                        filepath, line_n, timestamp, T_idx, frame[T_idx]))
                    for idx in range(-3, 3 + 1):
                        try:
                            print("bit {}: {}".format(
                                T_idx-idx, frame[T_idx-idx]))
                        except:
                            pass
                    return line_n
        frames = np.array(frames)
        # the array needs rotating 90 CW
        frames = np.rot90(frames, k=-1, axes=(1, 2))
    return -1
