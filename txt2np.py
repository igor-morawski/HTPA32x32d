import numpy as np
import os
import cv2
import imageio
import argparse
from tools import txt2np

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='.txt file to parse')
    parser.add_argument('--output', type=str, help='.np file to output parsed array')
    parser.add_argument('--gif', type=str, help='.gif output file, optional')
    parser.add_argument('--rot90', type=int, help='rotate [input]-times 90 degree')
    FLAGS, unparsed = parser.parse_known_args()
    if not FLAGS.input or not FLAGS.output:
        raise ValueError
    array, _ = txt2np(FLAGS.input)
    if FLAGS.rot90:
        array = np.rot90(array, k=FLAGS.rot90)
    normalized_sequence = (array-array.min())/(array.max()-array.min())
    np.save(FLAGS.output, array)
    if FLAGS.gif:
        array2gif(normalized_sequence, FLAGS.gif)
