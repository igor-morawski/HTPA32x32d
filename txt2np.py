import numpy as np
import os
import cv2
import imageio

DTYPE = 'float32'
def txt2np_C(filepath: str, array_size=32):
    '''
    returns frames [frame, height, width]
    temperatures are in grad C
    '''
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
                frame = split[0:1024]
                timestamp = split[-1][:-1]
                frame = np.array([int(T) for T in frame], dtype=DTYPE)
                frame = frame.reshape([array_size, array_size], order='F')
                frame *= 1e-2
                frames.append(frame)
                timestamps.append(timestamp)
        frames = np.array(frames)
    return frames, timestamps

def array2gif(array, filepath2save: str, fps=10):
    temperature = array 
    gray = (255 * temperature).astype(np.uint8)

    def heatmap(sequence, cv_colormap: int = cv2.COLORMAP_JET):
        heatmap_flat = cv2.applyColorMap(sequence.flatten(), cv_colormap)
        return heatmap_flat.reshape(sequence.shape + (3,))

    bgr = heatmap(gray)
    with imageio.get_writer(filepath2save, mode='I', duration=1/fps) as writer:
        for frame in bgr:
            writer.append_data(frame[:, :, ::-1])
    return

if __name__ == "__main__":
    a, b = txt2np_C(os.path.join("examples", "ex2.txt"))
    normalized_sequence = (a-a.min())/(a.max()-a.min())
    array2gif(normalized_sequence, os.path.join("examples", "ex2.gif"))
