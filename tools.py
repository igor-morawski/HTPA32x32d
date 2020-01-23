"""
Tools for Heimann HTPA recordings. 

Data types:
    * txt - Heimann HTPA recordings in *.TXT,
    * np - NumPy array of thermopile sensor array data, shaped [frames, height, width],
    * csv - thermopile sensor array data in *.csv format, NOT a pandas dataframe!
    * df - pandas dataframe
    * pc - NumPy array of pseudocolored thermopile sensor array data, shaped [frames, height, width, channels]
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
# TODO: 2 -> write, if not conversion


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


def npWriteCsv(output_fp: str, array, timestamps: list) -> bool:
    """
    Convert and save Heimann HTPA NumPy array shaped [frames, height, width] to .CSV dataframe

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


def readCsv2np(csv_fp: str):
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
    array = reshapeFlattenedFrames(array)
    return array, timestamps


def applyHeatmap(array, cv_colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
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
    Applies pseudocoloring (heatmap) to a sequence of thermal distribution. Same as applyHeatmap().
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
    return applyHeatmap(array, cv_colormap)


# TODO heatmap -> pc


def saveFrames(array, dir_name: str, extension: str = ".bmp") -> bool:
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
    for idx, frame in enumerate(array):
        cv2.imwrite(os.path.join(dir_name, "%d" % idx + extension), frame)
    return True


def flattenFrames(array):
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


def pcWriteGif(array, fp: str, fps=10, loop: int = 0, duration=None):
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


def timestamps2framedurations(timestamps: list, lastFrameDuration=None) -> list:
    """
    # TODO 
    Produces frame durations list to produce more accurate 
    # TODO 
    Parameters
    ----------
    array : np.array
         flattened array (frames, height, width)

    Returns
    -------
    np.array
        reshaped array (frames, height, width)
    """
    pass  # TODO


def reshapeFlattenedFrames(array):
    """
    Reshapes array shaped [frames, height*width] into array of shape [frames, height, width]

    Parameters
    ----------
    array : np.array
         flattened array (frames, height, width)

    Returns
    -------
    np.array
        reshaped array (frames, height, width)
    """
    _, elements = array.shape
    height = int(elements ** (1 / 2))
    width = height
    return array.reshape((-1, height, width))


if __name__ == "__main__":
    import os

    SAMPLE_FP = os.path.join("testing", "sample.TXT")
    SAMPLE_FP = os.path.join("examples", "person1.TXT")
    array, timestamps = txt2np(filepath=SAMPLE_FP, array_size=32)
    npWriteCsv(os.path.join("tmp", "try.csv"), array, timestamps)
    output_fp = os.path.join("tmp", "try.csv")
    frame = array[1, ...]
