from txt2np import txt2np_C
from txt2np import array2gif

import numpy as np
import os
import cv2
import imageio

import matplotlib
import matplotlib.pyplot as plt

example_dir = "examples" 
fn = "person1.TXT"
fp = os.path.join(example_dir, fn)

FPS = 8

n=99

def array2gif_gray(array, filepath2save: str, fps=10):
    temperature = array 
    gray = (255 * temperature).astype(np.uint8)
    def gray2bgr(sequence):
        heatmap_flat = cv2.cvtColor(sequence.flatten(), cv2.COLOR_GRAY2BGR)
        return heatmap_flat.reshape(sequence.shape + (3,))
    bgr = gray2bgr(gray)
    with imageio.get_writer(filepath2save, mode='I', duration=1/fps) as writer:
        for frame in bgr:
            writer.append_data(frame[:, :, ::-1])

if __name__ == "__main__":
    temperature, timestamps = txt2np_C(fp)
    #visualize n-th frame
    frame = temperature[n]
    fig, ax = plt.subplots()
    im = ax.imshow(frame, cmap='gray')
    cbar = ax.figure.colorbar(im, ax=ax) 
    plt.axis('off')
    plt.title('Temperature distribution')
    cbar.ax.set_ylabel('                  Temp. °C', rotation=0)
    plt.show()
    #normalization
    normalized_sequence = (temperature-temperature.min())/(temperature.max()-temperature.min())
    #visualize the normalized sequence
    array2gif_gray(normalized_sequence, os.path.join("examples", "person1_gray.gif"), fps=FPS)
    #and the heatmap
    array2gif(normalized_sequence, os.path.join("examples", "person1.gif"), fps=FPS)