
"""Convert Heimann HTPA .txt files to .csv (compatible with:
https://github.com/muralab/Low-Resolution-FIR-Action-Dataset)
If you need converting txt2cvs use tools module, this is just an example
"""

import numpy as np
import os
import argparse

from tools import txt2np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='.txt file to parse')
    parser.add_argument('--output', type=str, help='.csv file to output parsed array')
    