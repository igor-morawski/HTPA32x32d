import unittest
import numpy as np
import pandas as pd
import os
import cv2

import tools

EXPECTED_TXT_FP = os.path.join("testing", "expected.TXT")
EXPECTED_NP_FP = os.path.join("testing", "expected.npy")
EXPECTED_CSV_FP = os.path.join("testing", "expected.csv")
EXPECTED_PICKLE_FP = os.path.join("testing", "expected.pickle")
TMP_PATH = os.path.join("testing", "TMP")


def _init():
    if not os.path.exists(TMP_PATH):
        os.mkdir(TMP_PATH)


def _cleanup(files_fp: list = None):
    if files_fp:
        for fp in files_fp:
            os.remove(fp)
    if os.path.exists(TMP_PATH):
        os.rmdir(TMP_PATH)



class TestTxt2np(unittest.TestCase):
    def test_Result(self):
        expected_frames_shape = (3, 32, 32)
        expected_timestamps = [170.093, 170.218, 170.343]
        result = tools.txt2np(filepath=EXPECTED_TXT_FP, array_size=32)
        frames, timestamps = result
        self.assertEqual(timestamps, expected_timestamps)
        self.assertEqual(frames.shape, expected_frames_shape)
        expected_frames = np.load(EXPECTED_NP_FP)
        self.assertTrue(np.array_equal(frames, expected_frames))

    def test_Defaults(self):
        """
        Default tested: array_size = 32
        """
        expected = tools.txt2np(filepath=EXPECTED_TXT_FP, array_size=32)
        result = tools.txt2np(filepath=EXPECTED_TXT_FP)
        expected_frames, _ = expected
        frames, _ = result
        self.assertTrue(np.array_equal(frames, expected_frames))

    def test_DTYPE(self):
        expected_dtype = np.dtype(tools.DTYPE)
        result = tools.txt2np(filepath=EXPECTED_TXT_FP)
        frames, _ = result
        self.assertEqual(frames.dtype, expected_dtype)

    def test_array_size(self):
        expected_frames_shape = (3, 16, 16)
        expected_timestamps = [170.093, 170.218, 170.343]
        result = tools.txt2np(filepath=EXPECTED_TXT_FP, array_size=16)
        frames, timestamps = result
        self.assertEqual(timestamps, expected_timestamps)
        self.assertEqual(frames.shape, expected_frames_shape)


class Testwrite_np2csv(unittest.TestCase):
    def test_Result(self):
        expected_array = np.load(EXPECTED_NP_FP)
        timestamps = [170.093, 170.218, 170.343]
        csv_fp = os.path.join(TMP_PATH, "file.csv")
        tools.write_np2csv(csv_fp, expected_array, timestamps)
        self.assertTrue(os.path.exists(csv_fp))
        df = pd.read_csv(csv_fp, **tools.READ_CSV_ARGS)
        timestamps = df[tools.PD_TIME_COL]
        array = df.drop([tools.PD_TIME_COL, tools.PD_PTAT_COL], axis=1).to_numpy(
            dtype=tools.DTYPE
        )
        array = tools.reshape_flattened_frames(array)
        self.assertTrue(np.array_equal(array, expected_array))
        _cleanup([csv_fp])


class Test_csv2np(unittest.TestCase):
    def test_Result(self):
        expected_array = np.load(EXPECTED_NP_FP)
        expected_timestamps = [170.093, 170.218, 170.343]
        array, timestamps = tools.csv2np(EXPECTED_CSV_FP)
        self.assertTrue(np.array_equal(array, expected_array))
        self.assertTrue(np.array_equal(timestamps, expected_timestamps))


class Testwrite_np2pickle(unittest.TestCase):
    def test_Result(self):
        array = np.load(EXPECTED_NP_FP)
        timestamps = [170.093, 170.218, 170.343]
        pickle_fp = os.path.join(TMP_PATH, "file.pickle")
        tools.write_np2pickle(pickle_fp, array, timestamps)
        self.assertTrue(os.path.isfile(pickle_fp))
        _cleanup([pickle_fp])


class Testwrite_pc2gif(unittest.TestCase):
    def test_Defaults(self):
        gif_fp = os.path.join(TMP_PATH, "tmp.gif")
        temperature_array = np.load(EXPECTED_NP_FP)
        pc_array = tools.np2pc(temperature_array)
        tools.write_pc2gif(pc_array, gif_fp)
        self.assertTrue(os.path.exists(gif_fp))
        _cleanup([gif_fp])

    def test_DurationAndLoop(self):
        _init()
        gif_fp = os.path.join(TMP_PATH, "tmp.gif")
        height, width = 100, 100
        zeros = np.zeros([height, width], dtype=np.uint8)
        ones = 255 * np.ones([height, width], dtype=np.uint8)
        b = np.stack([ones, zeros, zeros], axis=-1)
        g = np.stack([zeros, ones, zeros], axis=-1)
        r = np.stack([zeros, zeros, ones], axis=-1)
        pc_array = np.stack([b, g, r])
        expected_timestamps = [1.424, 2.453453, 3.5345]
        duration_list = [
            x_t2 - x_t1
            for x_t1, x_t2 in zip(expected_timestamps, expected_timestamps[1:])
        ]
        duration_list.append(duration_list[-1])
        tools.write_pc2gif(pc_array, gif_fp, duration=duration_list, loop=1)
        self.assertTrue(os.path.exists(gif_fp))
        _cleanup([gif_fp])


class Test_timestamps2frame_durations(unittest.TestCase):
    def test_Result_Defaults(self):
        test = [1, 2, 4]
        expected_result = [1, 2, 2]
        result = tools.timestamps2frame_durations(test)
        self.assertEqual(expected_result, result)

    def test_Result_Op(self):
        test = [1, 2, 4]
        expected_result = [1, 2, 5]
        result = tools.timestamps2frame_durations(test, last_frame_duration=5)
        self.assertEqual(expected_result, result)


class Testflatten_frames(unittest.TestCase):
    def test_Result(self):
        input_shape = (100, 32, 32)
        expected_shape = (100, 32 * 32)
        input = np.arange(np.prod(input_shape)).reshape(input_shape)
        result = tools.flatten_frames(input)
        self.assertEqual(result.shape, expected_shape)


class Testapply_heatmap(unittest.TestCase):
    def test_Result(self):
        array = np.ones([2, 40, 40])
        array[0, :, :] = np.zeros([40, 40])
        # default color maps is cv2.:COLORMAP_JET
        result = tools.apply_heatmap(array)
        self.assertTrue(np.array_equal(result[0, 0, 0], [128, 0, 0]))
        self.assertTrue(np.array_equal(result[1, 0, 0], [0, 0, 128]))

    def test_Colormaps(self):
        array = np.ones([2, 40, 40])
        array[0, :, :] = np.zeros([40, 40])
        # cv2.COLORMAP_HOT = 11
        result = tools.apply_heatmap(array, cv_colormap=11)
        self.assertTrue(np.array_equal(result[0, 0, 0], [0, 0, 0]))
        self.assertTrue(np.array_equal(result[1, 0, 0], [255, 255, 255]))


class TestNp2pc(unittest.TestCase):
    def test_Result(self):
        array = np.ones([2, 40, 40])
        array[0, :, :] = np.zeros([40, 40])
        # default color maps is cv2.:COLORMAP_JET
        result = tools.np2pc(array)
        self.assertTrue(np.array_equal(result[0, 0, 0], [128, 0, 0]))
        self.assertTrue(np.array_equal(result[1, 0, 0], [0, 0, 128]))

    def test_Colormaps(self):
        array = np.ones([2, 40, 40])
        array[0, :, :] = np.zeros([40, 40])
        # cv2.COLORMAP_HOT = 11
        result = tools.np2pc(array, cv_colormap=11)
        self.assertTrue(np.array_equal(result[0, 0, 0], [0, 0, 0]))
        self.assertTrue(np.array_equal(result[1, 0, 0], [255, 255, 255]))


class Testsave_frames(unittest.TestCase):
    def test_Files(self):
        _init()
        height, width = 100, 100
        zeros = np.zeros([height, width], dtype=np.uint8)
        ones = 255 * np.ones([height, width], dtype=np.uint8)
        b = np.stack([ones, zeros, zeros], axis=-1)
        g = np.stack([zeros, ones, zeros], axis=-1)
        r = np.stack([zeros, zeros, ones], axis=-1)
        array = np.stack([b, g, r])
        tools.save_frames(array, TMP_PATH)
        fp_list = []
        for idx in range(3):
            fp = os.path.join(TMP_PATH, str(idx) + ".bmp")
            fp_list.append(fp)
            self.assertTrue(os.path.exists(fp))
            img = cv2.imread(fp)
            self.assertTrue(np.array_equal(img, array[idx]))
        _cleanup(fp_list)


class Testreshape_flattened_frames(unittest.TestCase):
    def test_Result(self):
        input_shape = (100, 32 * 32)
        expected_shape = (100, 32, 32)
        input = np.arange(np.prod(input_shape)).reshape(input_shape)
        result = tools.reshape_flattened_frames(input)
        self.assertEqual(result.shape, expected_shape)


class TestreshapingFrames(unittest.TestCase):
    def testflattenAndReshape(self):
        input = np.load(EXPECTED_NP_FP)
        flattened_result = tools.flatten_frames(input)
        first_frame = flattened_result[0]
        expected_first_frame = input[0].flatten()
        self.assertTrue(np.array_equal(first_frame, expected_first_frame))
        reshaped_result = tools.reshape_flattened_frames(flattened_result)
        self.assertTrue(np.array_equal(input, reshaped_result))


class Test_crop_center(unittest.TestCase):
    def test_shapes(self):
        a1 = np.zeros((5, 32, 32))
        a2 = np.zeros((5, 31, 31))
        result_2D_shape = (26, 26)
        a1_result = tools.crop_center(a1, *result_2D_shape)
        a2_result = tools.crop_center(a2, *result_2D_shape)
        self.assertEqual(a1_result.shape[1:], result_2D_shape)
        self.assertEqual(a2_result.shape[1:], result_2D_shape)
        a3 = np.zeros((5, 37, 32))
        a4 = np.zeros((5, 40, 50))
        result2_2D_shape = (4, 7)
        a3_result = tools.crop_center(a3, *result2_2D_shape)
        a4_result = tools.crop_center(a4, *result2_2D_shape)
        self.assertEqual(a3_result.shape[1:], result2_2D_shape)
        self.assertEqual(a4_result.shape[1:], result2_2D_shape)

    def test_result(self):
        array_shape = (2, 6, 6)
        array = np.arange(np.prod(array_shape)).reshape(array_shape)
        result_2D_shape = (4, 4)
        result = tools.crop_center(array, *result_2D_shape)
        expected_result = np.array([
            [[7, 8, 9, 10],
             [13, 14, 15, 16],
             [19, 20, 21, 22],
             [25, 26, 27, 28]],

            [[43, 44, 45, 46],
             [49, 50, 51, 52],
             [55, 56, 57, 58],
             [61, 62, 63, 64]]
        ])
        self.assertTrue(np.array_equal(result, expected_result))

    def test_no_crop(self):
        array_shape = (3, 32, 53)
        array = np.arange(np.prod(array_shape)).reshape(array_shape)
        result_2D_shape = array_shape[1:]
        result = tools.crop_center(array, *result_2D_shape)
        expected_result = array
        self.assertTrue(np.array_equal(result, expected_result))

    def test_minus1_crop(self):
        array_shape = (3, 32, 53)
        height = array_shape[1]
        width = array_shape[2]
        array = np.arange(np.prod(array_shape)).reshape(array_shape)
        result1 = tools.crop_center(array, height, width)
        result2 = tools.crop_center(array, -1, width)
        result3 = tools.crop_center(array, height, -1)
        result4 = tools.crop_center(array, -1, -1)
        expected_result = array
        self.assertTrue(np.array_equal(result1, expected_result))
        self.assertTrue(np.array_equal(result2, expected_result))
        self.assertTrue(np.array_equal(result3, expected_result))
        self.assertTrue(np.array_equal(result4, expected_result))
    

class Test_match_timesteps(unittest.TestCase):
    def test_3_lists(self):
        ts1 = [1, 2, 3, 4, 5]
        ts2 = [1.1, 2.1, 2.9, 3.6, 5.1, 6, 6.1]
        ts3 = [0.9, 1.2, 2, 3, 4.1, 4.2, 4.3, 4.9]
        results = tools.match_timesteps(ts1, ts2, ts3)
        expected_results = [None] * 3
        expected_results[0] = [0, 1, 2, 3, 4]
        expected_results[1] = [0, 1, 2, 3, 4]
        expected_results[2] = [0, 2, 3, 4, 7]
        self.assertEqual(results, expected_results)
        results = tools.match_timesteps(ts2, ts1, ts3)
        expected_results[0] = [0, 1, 2, 3, 4]
        expected_results[1] = [0, 1, 2, 3, 4]
        expected_results[2] = [0, 2, 3, 4, 7]
        self.assertEqual(results, expected_results)
        results = tools.match_timesteps(ts3, ts1, ts2, ts3, ts2)
        expected_results = [None] * 5
        expected_results[0] = [0, 2, 3, 4, 7]
        expected_results[1] = [0, 1, 2, 3, 4]
        expected_results[2] = [0, 1, 2, 3, 4]
        expected_results[3] = [0, 2, 3, 4, 7]
        expected_results[4] = [0, 1, 2, 3, 4]
        self.assertEqual(results, expected_results)

class Test_resample_np_tuples(unittest.TestCase):
    def test_indices(self):
        a1 = np.arange(4*10).reshape((-1,2,2))
        a2 = np.arange(4*10).reshape((-1,2,2))
        a3 = np.arange(4*10).reshape((-1,2,2))*5
        a_tuple = [a1, a2, a3]
        indices = [[0, 2, 5], [0, 1, 3], [0, 1, 3]]
        result = tools.resample_np_tuples(a_tuple, indices=indices)
        expected_result1 = np.array([[[0, 1], [2, 3]], [[8, 9], [10, 11]], [[20, 21], [22, 23]]])
        expected_result2 = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[12, 13], [14, 15]]])
        expected_result3 = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[12, 13], [14, 15]]]) * 5
        expected_result = [expected_result1, expected_result2, expected_result3]
        self.assertEqual(len(result), len(expected_result))
        for array, expected_array in zip(result, expected_result):
            self.assertTrue(np.array_equal(array, expected_array))
    def test_step(self):
        a1 = np.arange(4*4).reshape((-1,2,2))
        a2 = np.arange(4*8).reshape((-1,2,2))
        a3 = np.arange(4*8).reshape((-1,2,2))*5
        a_tuple = [a1, a2, a3]
        result = tools.resample_np_tuples(a_tuple, step=2)
        expected_result1 = np.array([[[0, 1], [2, 3]], [[8, 9], [10, 11]]])
        expected_result2 = np.array([[[0, 1], [2, 3]], [[8, 9], [10, 11]], [[16, 17], [18, 19]], [[24, 25], [26, 27]]])
        expected_result3 = np.array([[[0, 1], [2, 3]], [[8, 9], [10, 11]], [[16, 17], [18, 19]], [[24, 25], [26, 27]]])*5
        expected_result = [expected_result1, expected_result2, expected_result3]
        self.assertEqual(len(result), len(expected_result))
        for array, expected_array in zip(result, expected_result):
            self.assertTrue(np.array_equal(array, expected_array))
    def test_none(self):
        a1 = np.arange(4*4).reshape((-1,2,2))
        a2 = np.arange(4*8).reshape((-1,2,2))
        a3 = np.arange(4*8).reshape((-1,2,2))*5
        a_tuple = [a1, a2, a3]
        result = tools.resample_np_tuples(a_tuple)
        expected_result = a_tuple
        self.assertEqual(len(result), len(expected_result))
        for array, expected_array in zip(result, expected_result):
            self.assertTrue(np.array_equal(array, expected_array))
        result = tools.resample_np_tuples(a_tuple, step=1)
        expected_result = a_tuple
        self.assertEqual(len(result), len(expected_result,))
        for array, expected_array in zip(result, expected_result):
            self.assertTrue(np.array_equal(array, expected_array))
        result = tools.resample_np_tuples(a_tuple, indices=[list(range(4)), list(range(8)), list(range(8))])
        expected_result = a_tuple
        self.assertEqual(len(result), len(expected_result,))
        for array, expected_array in zip(result, expected_result):
            self.assertTrue(np.array_equal(array, expected_array))

class Test_resample_timestamps(unittest.TestCase):
    def test_indices(self):
        ts1 = [1, 2, 3, 4, 5]
        ts2 = [1.1, 2.1, 2.9, 3.6, 5.1, 6, 6.1]
        ts3 = [0.9, 1.2, 2, 3, 4.1, 4.2, 4.3, 4.9]
        timestamps = [ts1, ts2, ts3]
        indices = [None] * 3
        indices[0] = [0, 1, 2, 3, 4]
        indices[1] = [0, 1, 2, 3, 4]
        indices[2] = [0, 2, 3, 4, 7]
        result = tools.resample_timestamps(timestamps, indices=indices)
        expected_result = [ts1, ts2[:5], [0.9,2.0,3.0,4.1,4.9]]
        self.assertEqual(result, expected_result)
    def test_step(self):
        ts1 = [1, 2, 3, 4, 5]
        ts2 = [1.1, 2.1, 2.9, 3.6, 5.1, 6, 6.1]
        ts3 = [0.9, 1.2, 2, 3, 4.1, 4.2, 4.3, 4.9]
        timestamps = [ts1, ts2, ts3]
        result = tools.resample_timestamps(timestamps, step=2)
        expected_result = [[1,3,5], [1.1, 2.9, 5.1, 6.1], [0.9,2, 4.1, 4.3]]
        self.assertEqual(result, expected_result)


class Test_read_tpa_file(unittest.TestCase):
    def test_txt(self):
        expected_array = np.load(EXPECTED_NP_FP)
        expected_timestamps = [170.093, 170.218, 170.343]
        array, timestamps = tools.read_tpa_file(EXPECTED_TXT_FP)
        self.assertTrue(np.array_equal(array, expected_array))
        self.assertTrue(np.array_equal(timestamps, expected_timestamps))
    def test_csv(self):
        expected_array = np.load(EXPECTED_NP_FP)
        expected_timestamps = [170.093, 170.218, 170.343]
        array, timestamps = tools.read_tpa_file(EXPECTED_CSV_FP)
        self.assertTrue(np.array_equal(array, expected_array))
        self.assertTrue(np.array_equal(timestamps, expected_timestamps))
    def test_pickle(self):
        expected_array = np.load(EXPECTED_NP_FP)
        expected_timestamps = [170.093, 170.218, 170.343]
        array, timestamps = tools.read_tpa_file(EXPECTED_PICKLE_FP)
        self.assertTrue(np.array_equal(array, expected_array))
        self.assertTrue(np.array_equal(timestamps, expected_timestamps))
