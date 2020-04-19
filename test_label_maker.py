import unittest
import numpy as np
import pandas as pd
import os
import cv2

import tools

import label_maker

EXPECTED_TXT_FP = os.path.join("testing", "expected.TXT")


def _init():
    if not os.path.exists(TMP_PATH):
        os.mkdir(TMP_PATH)


def _cleanup(files_fp: list = None):
    if files_fp:
        for fp in files_fp:
            os.remove(fp)
    if os.path.exists(TMP_PATH):
        os.rmdir(TMP_PATH)

class Test(unittest.TestCase):
    def test_Result(self):
        pass