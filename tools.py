import numpy as np
import pandas as pd
DTYPE = 'float32'
PD_SEP = ';'
PD_NAN = np.inf
PD_DTYPE = np.float32

def txt2np(filepath: str, array_size:int=32):
    """
    Convert Heimann HTPA .txt to NumPy array shaped [frame, height, width].

    Parameters
    ----------
    filepath : str
    array_size : int, optional

    Returns
    -------
    np.array
        3D array of temperature distribution sequence, shaped [frame, height, width].
    """
    with open(filepath) as f:
        #discard the first line
        _ = f.readline()
        #read line by line now
        line = "dummy line"
        frames = []
        timestamps = []
        while line:
            line = f.readline()
            if line:
                split = line.split(" ")
                frame = split[0:array_size**2]
                timestamp = split[-1]
                frame = np.array([int(T) for T in frame], dtype=DTYPE)
                frame = frame.reshape([array_size, array_size], order='F')
                frame *= 1e-2
                frames.append(frame)
                timestamps.append(float(timestamp))
        frames = np.array(frames)
        # the array needs rotating 90 CW
        frames = np.rot90(frames, k=-1, axes = (1, 2))
    return frames, timestamps

def np2csv(output_fp:str, array, timestamps: list):
    """
    Convert Heimann HTPA NumPy array shaped [frame, height, width] to .CSV dataframe

    Parameters
    ----------
    output_fp : str
        Filepath to destination file, including the file name.
    array : np.array
        Temperatue distribution sequence, shape [frame, height, width].
    timestamps : list
        List of timestamps of corresponding array frames.
    """
    # initialize csv template (and append frames later)
    # prepend first row for compability with legacy format
    first_row = pd.DataFrame({'HTPA 32x32d': []})
    first_row.to_csv(output_fp, index=False, sep=PD_SEP)
    headers = {'Time (sec)':[],'PTAT':[]}
    df = pd.DataFrame(headers)
    for idx in range(np.prod(array.shape[1:])):
        df.insert(len(df.columns), 'P%04d'%idx, [])
    # TODO Change separator to , instead of ;
    df.to_csv(output_fp, mode='a', index=False, sep=PD_SEP)

    for idx in range(array.shape[0]):
        frame = array[idx,...]
        timestamp = timestamps[idx]
        temps = list(frame.flatten())
        row_data = [timestamp, PD_NAN]
        row_data.extend(temps)
        row = pd.DataFrame([row_data])
        row = row.astype(PD_DTYPE)
        row.to_csv(output_fp, mode='a', header=False, sep=PD_SEP,index=False)
    return True

        


def _flattenFrames(array):
    """
    Flattens array of shape (frame, height, width) into array of shape (frame, height*width)

    Parameters
    ----------
    array : np.array
         (frame, height, width)

    Returns
    -------
    np.array
        flattened array
         (frame, height, width)
    """
    _, height, width = array.shape
    return array.reshape((-1, height*width))

def _reshapeFlattenedFrames(array):
    """
    Reshapes array of shape (frame, height*width) into array of shape (frame, height, width)

    Parameters
    ----------
    array : np.array
         flattened array (frame, height, width)

    Returns
    -------
    np.array
        reshaped array (frame, height, width)
    """
    _, elements = array.shape
    height = int(elements**(1/2))
    width = height
    return array.reshape((-1, height, width))

if __name__ == "__main__":
    import os
    SAMPLE_FP = os.path.join("testing", "sample.TXT")
    SAMPLE_FP = os.path.join("examples", "person1.TXT")
    array, timestamps = txt2np(filepath = SAMPLE_FP, array_size = 32)
    np2csv(os.path.join("tmp","try.csv"), array, timestamps)
    output_fp = os.path.join("tmp","try.csv")
    frame = array[1,...]

