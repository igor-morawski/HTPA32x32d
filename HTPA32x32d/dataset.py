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


import HTPA32x32d.tools as tools


VERBOSE = False


TPA_PREFIX_TEMPLATE = "YYYYMMDD_HHMM_ID{VIEW_IDENTIFIER}"
TPA_NFO_FN = "tpa.nfo"
PROCESSED_OK_KEY = "PROCESSED_OK"
MADE_OK_KEY = "MAKE_OK"
SYNCHRONIZATION_MAX_ERROR = 0.05
DATASET_POSITIVE_ONE_HOT = np.array([0, 1])
DATASET_NEGATIVE_ONE_HOT = np.array([1, 0])


class _TPA_Sample():
    """
    Use TPA_Sample_from_filepaths or TPA_Sample_from_data that inherit from this class.
    """

    def __init__(self, filepaths, ids, arrays, timestamps):
        self.filepaths = filepaths
        self.ids = ids
        self.arrays = arrays
        self.timestamps = timestamps

    def test_alignment(self):
        lengths = [len(ts) for ts in self.timestamps]
        return all(l == lengths[0] for l in lengths)

    def test_synchronization(self, max_error):
        pairs = itertools.combinations(self.timestamps, 2)
        for pair in pairs:
            if (np.abs(np.array(pair[0]) - np.array(pair[1])).max() > max_error):
                return False
        return True

    def write_gif(self):
        """
        Writes visualization gif to same directory as in self.filepaths,
        the filename follows the template: FILE_PREFIX_ID{id1}-{id2}-...-{idn}.gif
        """
        if not self.test_alignment():
            raise Exception("Unaligned sequences cannot be synchronized!")
        data = np.concatenate(self.arrays, axis=2)
        pc = tools.np2pc(data)
        ts = np.sum(self.timestamps, axis=0)/len(self.timestamps)
        duration = tools.timestamps2frame_durations(ts)
        head, tail = os.path.split(self.filepaths[0])
        fn = _TPA_get_file_prefix(tail) + "ID" + "-".join(self.ids) + ".gif"
        fp = os.path.join(head, fn)
        tools.write_pc2gif(pc, fp, duration=duration)


class TPA_Sample_from_filepaths(_TPA_Sample):
    """
    Data structure for loading a mutli-view TPA sample from given filepaths.

    Attributes
    ----------
    filepaths : list
        Filepaths of files that sample was loaded from.
    ids : list
        List of ids corresponding to arrays and timestamps.
    arrays : list
        List of arrays (TPA sequences [frames, height, width]).
    timestamps : list
        List of lists of timestamps corresponding to each timestep.

    Methods
    -------
    test_synchronization(max_error)
        returns False if max_error exceeded at any timestep (units: [s]), True otherwise.
    test_alignment()
        returns True if arrays are the same length. 
    """

    def __init__(self, filepaths):
        ids = [self._read_ID(fp) for fp in filepaths]
        samples = [tools.read_tpa_file(fp) for fp in filepaths]
        arrays = [sample[0] for sample in samples]
        timestamps = [sample[1] for sample in samples]
        _TPA_Sample.__init__(self, filepaths, ids, arrays, timestamps)

    def _read_ID(self, filepath):
        fn = os.path.basename(filepath)
        name = tools.remove_extension(fn)
        return name.split("ID")[-1]

    def write(self):
        """
        Not implemUse TPA_Sample_from_data if you need to modify arrays.
        """
        raise Exception(
            "You are trying to overwrite files. Use TPA_Sample_from_data if you need to modify arrays.")

    def align_timesteps(self):
        """
        Not implemUse TPA_Sample_from_data if you need to modify arrays.
        """
        raise Exception(
            "Use TPA_Sample_from_data if you need to modify arrays.")

    def get_header(self):
        return tools.read_txt_header(self.filepaths[0])


class TPA_Sample_from_data(_TPA_Sample):
    """
    Data structure for loading a mutli-view TPA sample from given filepaths.

    Attributes
    ----------
    ids : list
        List of ids corresponding to arrays and timestamps.
    arrays : list
        List of arrays (TPA sequences [frames, height, width]).
    timestamps : list
        List of lists of timestamps corresponding to each timestep.
    filepaths : list, optional
        Filepaths to write arrays to when using write().

    Methods
    -------
    align_timesteps(reset_T0 = False)
        align arrays in time, refer to match_timesteps in this module for details.
    write()
        write arrays stored in self.arrays to filepaths in self.filepaths.
    test_synchronization(max_error)
        returns False if max_error exceeded at any timestep (units: [s]), True otherwise.
    test_alignment()
        returns True if arrays are the same length. 
    """

    def __init__(self, arrays, timestamps, ids, output_filepaths=None, header=None):
        filepaths = None
        ids = ids.copy()
        arrays = arrays.copy()
        timestamps = timestamps.copy()
        self.header = header
        _TPA_Sample.__init__(self, filepaths, ids, arrays, timestamps)
        if output_filepaths:
            self.filepaths = output_filepaths

    def make_filepaths(self, parent_dir, prefix, extension):
        self.filepaths = [os.path.join(
            parent_dir, prefix+"ID"+id+"."+extension) for id in self.ids]

    def write(self):
        """
        Write stored arrays to filepaths in self.filepaths
        """
        assert self.filepaths
        if self.header:
            assert (tools.get_extension(self.filepaths[0]).lower() == 'txt')
        for fp, array, ts in zip(self.filepaths, self.arrays, self.timestamps):
            tools.write_tpa_file(fp, array, ts, header=self.header)
        return True

    def align_timesteps(self, reset_T0=False):
        """
        Align timesteps. Refer to match_timesteps() in this module for details.

        Parameters
        ----------
        reset_T0 : bool, optional
            If True delay of the inital frame will be removed from timestamps
        """
        indexes = tools.match_timesteps(*self.timestamps)
        for i in range(len(self.ids)):
            self.arrays[i] = self.arrays[i][indexes[i]]
            timestamps = np.array(self.timestamps[i])[indexes[i]]
            self.timestamps[i] = list(timestamps)
        if reset_T0:
            sample_T0_min = np.min([ts[0] for ts in self.timestamps])
            timestamps = [np.array(ts)-sample_T0_min for ts in self.timestamps]
            self.timestamps = timestamps
        return True


def _TPA_get_file_prefix(filepath):
    name = tools.remove_extension(os.path.basename(filepath))
    return name.split("ID")[0]


class _TPA_File_Manager():
    """
    TPA_Preparer and TPA_Dataset_Maker inherit from this class.
    """

    def __init__(self, reset_log=True):
        self.configured = False
        if (VERBOSE and reset_log):
            self._make_log = "make.log"
            if os.path.exists(self._make_log):
                os.remove(self._make_log)
        self._log_msgs = []

    def _log(self, log_msg):
        if VERBOSE:
            print(log_msg)
            with open(self._make_log, 'a') as f:
                f.write(log_msg+"\n")

    def _generate_config_template(self, output_json_filepath, fill_dict=None):
        template = {}
        for key in self._json_required_keys:
            template[key] = ""
        if fill_dict:
            for key in fill_dict:
                template[key] = fill_dict[key]
        with open(output_json_filepath, 'w') as f:
            json.dump(template, f)
        return True

    def _validate_config(self):
        keys_missing = []
        for key in self._json_required_keys:
            try:
                self._json[key]
            except KeyError:
                keys_missing.append(key)
        if len(keys_missing):
            msg = "Keys required {}".format(self._json_required_keys)
            self._log(msg)
            msg = "Keys missing {}".format(keys_missing)
            self._log(msg)
            return False
        try:
            if (self._json["PREPARE"] and self._json["MAKE"]):
                raise Exception(
                    "MAKE and PREPARE flags cannot be set at the same time to TRUE")
        except KeyError:
            pass
        return True

    def _remove_missing_views(self, prefixes, prefixes2filter):
        # filter out samples that miss views
        counter = collections.Counter(prefixes)
        view_number = len(self.view_IDs)
        prefixes2filter_copy = prefixes2filter.copy()
        for prefix in counter.keys():
            prefix_view_number = counter[prefix]
            if (prefix_view_number < view_number):
                prefixes2filter_copy.remove(prefix)
                self._log('[WARNING] Ignoring prefix {} because it misses {} views'.format(
                    prefix, view_number-prefix_view_number))
        prefixes2filter = prefixes2filter_copy.copy()
        return prefixes2filter

    def _remove_missing_rgbs(self, prefixes2process: list, scanned_dir: str):
        rgb_dirs_prefixes = set([_TPA_get_file_prefix(dir) for dir in glob.glob(
            os.path.join(scanned_dir, "*IDRGB"))])
        prefixes2process_copy = prefixes2process.copy()
        for prefix in prefixes2process:
            if prefix not in rgb_dirs_prefixes:
                prefixes2process_copy.remove(prefix)
                self._log(
                    '[WARNING] Ignoring prefix {} because it misses RGB view'.format(prefix))
        return prefixes2process_copy


class _Preparer(_TPA_File_Manager):
    def __init__(self, reset_log=True):
        _TPA_File_Manager.__init__(self, reset_log)
        self._json_required_keys = ["raw_input_dir", "processed_destination_dir", "view_IDs",
                                    "tpas_extension", "MAKE", "PREPARE"]

    def generate_config_template(self, output_json_filepath):
        self._generate_config_template(
            output_json_filepath, {"MAKE": 0, "PREPARE": 1})

    def _config(self, json_filepath):
        with open(json_filepath) as f:
            self._json = json.load(f)
        assert self._validate_config()
        assert self._json["PREPARE"]
        self.raw_input_dir = self._json["raw_input_dir"]
        self.processed_destination_dir = self._json["processed_destination_dir"]
        self.view_IDs = self._json["view_IDs"]
        self.tpas_extension = self._json["tpas_extension"]
        try:
            self.visualize = bool(self._json['VISUALIZE'])
        except KeyError:
            self.visualize = False
        try:
            self.undistort = bool(self._json['UNDISTORT'])
        except KeyError:
            self.undistort = False
        try:
            self.calib_fp = self._json['calib_fp']
        except KeyError:
            self.calib_fp = None
        self.configured = True
        return True

    def _write_nfo(self):
        filepath = os.path.join(self.processed_destination_dir, TPA_NFO_FN)
        data = {PROCESSED_OK_KEY: 1}
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def _write_labels_file(self, prefixes2label, labels_dict = None):
        filepath = os.path.join(self.processed_destination_dir, "labels.json")
        data = {prefix: "" for prefix in prefixes2label}
        if labels_dict:
            for key in labels_dict:
                try:
                    data[key] = labels_dict[key]
                except KeyError:
                    pass
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def _write_make_file(self):
        filepath = os.path.join(
            self.processed_destination_dir, "make_config.json")
        dataset_maker = TPA_Dataset_Maker(reset_log=False)
        fill_dict = {}
        fill_dict.update({"view_IDs": self.view_IDs})
        fill_dict.update({"tpas_extension": self.tpas_extension})
        fill_dict.update(
            {"processed_input_dir": self.processed_destination_dir})
        fill_dict.update({"labels_filepath": os.path.join(
            self.processed_destination_dir, "labels.json")})
        fill_dict.update({"MAKE": 1})
        fill_dict.update({"PREPARE": 0})
        dataset_maker.generate_config_template(filepath, fill_dict)


class TPA_Preparer(_Preparer):
    """
    Prepare files by processing raw samples (frame alignment) and generating a label file to be 
    filled by user needed for dataset generation. 
    - unproceesed sequences → aligned sequences and labels file (to be filled by user before making dataset)
    - filtering out samples that miss views (incomplete sequences)
    - aligning sequences
    - set HTPA32x32d.tools.SYNCHRONIZATION_MAX_ERROR in [s] that you're willing to tollerate
    Call generate_config_template() method to generate required config files.

    Input: 
    *ID*.TXT
    {config_making}.json
    tpa.nfo
    labels.json

    Output:
    *ID*.TXT
    tpa.nfo

    Arguments
    ---------
    configured : bool
        True if TPA_Preparer is ready to use prepare() method. Configure by calling config()
    Methods 
    -------
    generate_config_template()
        Generate config file template (json) to be filled by user 
        and passed to config()
    config()
        Configure #TODO FINISH DOCS
    
    """

    def __init__(self):
        _Preparer.__init__(self)

    def config(self, json_filepath):
        self._config(json_filepath)
        if any([self.undistort, self.calib_fp]):
            self._log(
                "[WARNING] UNDISTORT and calib_fp not supported in TPA_Preparer")
        return True

    def prepare(self):
        if not self.configured:
            msg = "Configure with config() first"
            self._log(msg)
            raise Exception(msg)
        if not (self.raw_input_dir):
            msg = "Destination directory filepath not specified"
            self._log(msg)
            raise ValueError(msg)
        tools.ensure_path_exists(self.raw_input_dir)
        glob_patterns = [os.path.join(
            self.raw_input_dir, "*ID"+id+"."+self.tpas_extension) for id in self.view_IDs]
        files = []
        for pattern in glob_patterns:
            files.extend(glob.glob(pattern))
        prefixes = [_TPA_get_file_prefix(f) for f in files]
        prefixes2process = list(set(prefixes))
        prefixes2process_number0 = len(prefixes2process)
        # filter out samples that miss views
        prefixes2process = self._remove_missing_views(
            prefixes, prefixes2process)
        prefixes2process_number = len(set(prefixes2process))
        prefixes_ignored = prefixes2process_number0 - prefixes2process_number
        self._log("[INFO] {} prefixes ignored out of initial {}".format(
            prefixes_ignored, prefixes2process_number0))
        self._log('"VISUALIZE" set to {}'.format(self.visualize))
        self._log("Reading, aligning and removing T0 from samples...")
        QUIT = False
        for prefix in prefixes2process:
            raw_fp_prefix = os.path.join(self.raw_input_dir, prefix)
            processed_fp_prefix = os.path.join(
                self.processed_destination_dir, prefix)
            raw_fps = [raw_fp_prefix + "ID" + view_id + "." +
                       self.tpas_extension for view_id in self.view_IDs]
            processed_fps = [processed_fp_prefix + "ID" + view_id +
                             "." + self.tpas_extension for view_id in self.view_IDs]
            raw_sample = TPA_Sample_from_filepaths(raw_fps)
            processed_sample = TPA_Sample_from_data(
                raw_sample.arrays, raw_sample.timestamps, raw_sample.ids, processed_fps)
            processed_sample.align_timesteps(reset_T0=True)
            if not processed_sample.test_synchronization(max_error=SYNCHRONIZATION_MAX_ERROR):
                QUIT = True
                self._log("[ERROR] {} did not pass synchronization test (max error {} s exceeded)!".format(
                    prefix, SYNCHRONIZATION_MAX_ERROR))
                continue
            processed_sample.write()
            if self.visualize:
                processed_sample.write_gif()
        assert not QUIT
        self._write_nfo()
        self._write_labels_file(prefixes2process)
        self._write_make_file()
        self._log("Writing nfo, labels and json files...")
        self._log("OK")


class _Dataset_Maker(_TPA_File_Manager):
    def __init__(self, reset_log=True):
        _TPA_File_Manager.__init__(self, reset_log)
        self._json_required_keys = ["dataset_destination_dir", "view_IDs",
                                    "processed_input_dir", "labels_filepath", "tpas_extension", "MAKE", "PREPARE"]

    def generate_config_template(self, output_json_filepath, fill_dict=None):
        init_fill_dict = {"MAKE": 1, "PREPARE": 0}
        if fill_dict:
            init_fill_dict.update(fill_dict)
        self._generate_config_template(
            output_json_filepath, fill_dict=init_fill_dict)

    def _config(self, json_filepath):
        with open(json_filepath) as f:
            self._json = json.load(f)
        assert self._validate_config()
        assert self._json["MAKE"]
        self.dataset_destination_dir = self._json["dataset_destination_dir"]
        self.view_IDs = self._json["view_IDs"]
        self.processed_input_dir = self._json["processed_input_dir"]
        if not os.path.exists(os.path.join(self.processed_input_dir, TPA_NFO_FN)):
            raise Exception("{} doesn't exist. Process your data first using TPA_Preparer".format(
                os.path.join(self.processed_input_dir, TPA_NFO_FN)))
        with open(os.path.join(self.processed_input_dir, TPA_NFO_FN)) as f:
            nfo = json.load(f)
        assert nfo[PROCESSED_OK_KEY]
        self.labels_filepath = self._json["labels_filepath"]
        if not os.path.exists(self._json["labels_filepath"]):
            msg = "Specified label file doesn't exist {}".format(
                self._json["labels_filepath"])
            self._log(msg)
            raise Exception(msg)
        self.tpas_extension = self._json["tpas_extension"]
        self.configured = True
        return True

    def _write_nfo(self):
        filepath = os.path.join(self.dataset_destination_dir, TPA_NFO_FN)
        data = {MADE_OK_KEY: 1}
        data["view_IDs"] = self.view_IDs
        data["tpas_extension"] = self.tpas_extension
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def _copy_labels_file(self):
        src = os.path.join(self.labels_filepath)
        dst = os.path.join(self.dataset_destination_dir, "labels.json")
        shutil.copy2(src, dst)

    def _read_labels_file(self, json_filepath):
        with open(json_filepath) as f:
            data = json.load(f)
        for key in data.keys():
            if (len(key.split("ID")) > 1):
                old_key = key
                new_key = key.split("ID")[0]
                data[new_key] = data.pop(old_key)
        return data


class TPA_Dataset_Maker(_Dataset_Maker):
    '''
    CALL A TPA_Preparer FIRST ; #TODO FINISH DOCS
    '''

    def __init__(self, reset_log=True):
        _Dataset_Maker.__init__(self, reset_log)

    def config(self, json_filepath):
        self._config(json_filepath)
        return True

    def make(self):
        if not self.configured:
            msg = "Configure with config() first"
            self._log(msg)
            raise Exception(msg)
        if not (self.dataset_destination_dir):
            msg = "Destination directory filepath not specified"
            self._log(msg)
            raise ValueError(msg)
        tools.ensure_path_exists(self.dataset_destination_dir)
        glob_patterns = [os.path.join(
            self.processed_input_dir, "*ID"+id+"."+self.tpas_extension) for id in self.view_IDs]
        files = []
        for pattern in glob_patterns:
            files.extend(glob.glob(pattern))
        prefixes = [_TPA_get_file_prefix(f) for f in files]
        prefixes2make = list(set(prefixes))
        prefixes2make_number0 = len(prefixes2make)
        # filter out samples that miss views
        prefixes2make = self._remove_missing_views(prefixes, prefixes2make)
        # filter out samples that miss a label
        self._labels = self._read_labels_file(self.labels_filepath)
        prefixes2make_copy = prefixes2make.copy()
        for prefix in prefixes2make:
            if (prefix not in self._labels):
                prefixes2make_copy.remove(prefix)
                self._log('[WARNING] Ignoring prefix {} because it misses a label'.format(
                    prefix))
        prefixes2make = prefixes2make_copy.copy()
        for prefix in prefixes2make:
            if not self._labels[prefix]:
                prefixes2make_copy.remove(prefix)
                self._log('[WARNING] Ignoring prefix {} because it misses a label'.format(
                    prefix))
        prefixes2make = prefixes2make_copy.copy()
        for prefix in prefixes2make:
            if not (type(self._labels[prefix]) == int):
                prefixes2make_copy.remove(prefix)
                self._log('[WARNING] Ignoring prefix {} because the label is incorrect (it is not an integer)'.format(
                    prefix))
        prefixes2make = prefixes2make_copy.copy()
        # process the files
        fps2copy = []
        fps2output = []
        for prefix in prefixes2make:
            fp_prefix = os.path.join(self.processed_input_dir, prefix)
            fp_o_prefix = os.path.join(self.dataset_destination_dir, prefix)
            fps = []
            fps_o = []
            for view_id in self.view_IDs:
                fp = fp_prefix + "ID" + view_id + "." + self.tpas_extension
                fp_o = fp_o_prefix + "ID" + view_id + "." + self.tpas_extension
                fps.append(fp)
                fps_o.append(fp_o)
            fps2copy.append(fps)
            fps2output.append(fps_o)
        prefixes2make_number = len(set(prefixes2make))
        prefixes_ignored = prefixes2make_number0 - prefixes2make_number
        self._log("[INFO] {} prefixes ignored out of initial {}".format(
            prefixes_ignored, prefixes2make_number0))
        if (prefixes_ignored == prefixes2make_number0):
            self._log("[WARNING] All files ignored, the dataset is empty.")
            self._log("FAILED")
            return False
        self._log("[INFO] Making dataset...")
        self._log("[INFO] Copying files...")
        for src_tuple, dst_tuple in zip(fps2copy, fps2output):
            for src, dst in zip(src_tuple, dst_tuple):
                shutil.copy2(src, dst)
        self._log("Writing nfo, labels and json files...")
        self._write_nfo()
        self._copy_labels_file()
        self._log("OK")
        return True

### TPA (multi-view) + RGB (one-view) samples


class _TPA_RGB_Sample():
    """
    Use TPA_RGB_Sample_from_filepaths or TPA_RGB_Sample_from_data that inherit from this class.
    """

    def __init__(self, TPA, RGB):
        #def __init__(self, filepaths, ids, arrays, timestamps, rgb_file_list, rgb_timestamps):
        self.TPA = TPA
        self.RGB = RGB
        self._update_TPA_RGB_timestamps()

    def _update_TPA_RGB_timestamps(self):
        self._TPA_RGB_timestamps = self.TPA.timestamps + [self.RGB.timestamps]

    def test_alignment(self):
        lengths = [len(ts) for ts in self._TPA_RGB_timestamps]
        return all(l == lengths[0] for l in lengths)

    def test_synchronization(self, max_error):
        pairs = itertools.combinations(self._TPA_RGB_timestamps, 2)
        for pair in pairs:
            if (np.abs(np.array(pair[0]) - np.array(pair[1])).max() > max_error):
                return False
        return True

    def read_rgb_timesteps(self, filepath: str):
        """
        #TODO DOCS
        """
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
        return result

    def write_gif(self):  
        """
        Writes visualization gif to same directory as in self.filepaths,
        the filename follows the template: FILE_PREFIX_ID{id1}-{id2}-...-{idn}.gif
        """
        if not self.test_alignment():
            raise Exception("Unaligned sequences cannot be synchronized!")
        data = np.concatenate(self.TPA.arrays, axis=2)
        pc = tools.np2pc(data)
        rgb_height, rgb_width = (cv2.imread(self.RGB.filepaths[0]).shape)[0:2]
        # 
        pc = np.insert(pc, range(pc.shape[2]//len(self.TPA.arrays), pc.shape[2], pc.shape[2]//len(self.TPA.arrays)), 0, axis=2)
        old_h, old_w = pc.shape[1:3]
        new_width = int((rgb_width/old_w)*old_w)
        new_height = int((rgb_width/old_w)*old_h)
        #
        pc_reshaped = [cv2.resize(frame, dsize=(
            new_width, new_height), interpolation=cv2.INTER_NEAREST) for frame in pc]
        pc_reshaped = np.array(pc_reshaped).astype(np.uint8)
        margin_size = rgb_width-new_width
        pc_frames, pc_height, pc_width, pc_ch = pc_reshaped.shape
        pc = np.concatenate([pc_reshaped, np.zeros(
            [pc_frames, pc_height, margin_size, pc_ch], dtype=np.uint8)], axis=2)
        img_sequence = [cv2.imread(fp) for fp in self.RGB.filepaths]
        rgb_sequence = np.array(img_sequence).astype(np.uint8)
        vis = np.concatenate([pc, rgb_sequence], axis=1)
        ts = np.sum(self._TPA_RGB_timestamps, axis=0) / \
            len(self._TPA_RGB_timestamps)
        duration = tools.timestamps2frame_durations(ts)
        head, tail = os.path.split(self.TPA.filepaths[0])
        fn = _TPA_get_file_prefix(tail) + "ID" + \
            "-".join(self.TPA.ids) + "-RGB" + ".gif"
        fp = os.path.join(head, fn)
        tools.write_pc2gif(vis, fp, duration=duration)


class RGB_Sample_from_filepaths():
    def __init__(self, rgb_directory):
        if os.path.exists(os.path.join(rgb_directory, "timesteps.pkl")):
            with open(os.path.join(rgb_directory, 'timesteps.pkl'), 'rb') as f:
                filepaths = pickle.load(f)
            self.filepaths = [os.path.join(rgb_directory, fn)
                              for fn in filepaths]
            self.timestamps = [float(tools.remove_extension(
                os.path.basename(fp)).replace("-", ".")) for fp in filepaths]
        else:
            globbed_rgb_dir = list(
                glob.glob(os.path.join(rgb_directory, "*-*[0-9]." + tools.HTPA_UDP_MODULE_WEBCAM_IMG_EXT)))
            if not globbed_rgb_dir:
                raise ValueError(
                    "Specified directory {} is empty or doesn't exist.".format(rgb_directory))
            unsorted_timestamps = [float(tools.remove_extension(
                os.path.basename(fp)).replace("-", ".")) for fp in globbed_rgb_dir]
            self.timestamps, self.filepaths = (list(t) for t in zip(
                *sorted(zip(unsorted_timestamps, globbed_rgb_dir))))


class TPA_RGB_Sample_from_filepaths(_TPA_RGB_Sample):
    """
    #TODO
    """

    def __init__(self, tpa_filepaths, rgb_directory):
        TPA = TPA_Sample_from_filepaths(tpa_filepaths)
        RGB = RGB_Sample_from_filepaths(rgb_directory)
        _TPA_RGB_Sample.__init__(self, TPA, RGB)
        if os.path.exists(os.path.join(rgb_directory, "label.txt")):
            with open(os.path.join(rgb_directory, "label.txt"), "r") as f:
                data = f.read()
            self.label = int(data.strip())

    def get_header(self):
        return tools.read_txt_header(self.TPA.filepaths[0])

    def write(self):
        """
        Use TPA_Sample_from_data if you need to modify arrays.
        """
        raise Exception(
            "Use TPA_RGB_Sample_from_data if you need to modify arrays.")

    def align_timesteps(self):
        """
        Use TPA_Sample_from_data if you need to modify arrays.
        """
        raise Exception(
            "Use TPA_RGB_Sample_from_data if you need to modify arrays.")


class TPA_RGB_Sample_from_data(_TPA_RGB_Sample):
    """
    #TODO
    TPA is from data
    RGB is from filepaths
    """

    def __init__(self, tpa_arrays, tpa_timestamps, tpa_ids, rgb_directory, tpa_output_filepaths=None, rgb_output_directory=None, header=None):
        TPA = TPA_Sample_from_data(
            tpa_arrays, tpa_timestamps, tpa_ids, tpa_output_filepaths, header=header)
        RGB = RGB_Sample_from_filepaths(rgb_directory)
        self.rgb_output_directory = rgb_output_directory
        _TPA_RGB_Sample.__init__(self, TPA, RGB)

    def write(self):
        """
        Write stored arrays to filepaths in self.TPA.filepaths
        and RGB bitmaps to self.rgb_output_directory
        """
        assert self.test_alignment()
        assert self.TPA.filepaths
        assert self.rgb_output_directory
        self.TPA.write()
        if not (self.rgb_output_directory):
            msg = "Destination directory filepath not specified"
            self._log(msg)
            raise ValueError(msg)
        tools.ensure_path_exists(self.rgb_output_directory)
        dst_filepaths = []
        for src, timestamp in zip(self.RGB.filepaths, self.RGB.timestamps):
            new_fn = "{:.2f}".format(timestamp).replace(
                ".", "-") + "." + tools.HTPA_UDP_MODULE_WEBCAM_IMG_EXT
            dst = os.path.join(self.rgb_output_directory, new_fn)
            shutil.copy2(src, dst)
            dst_filepaths.append(dst)
        '''
        write filepaths (relative to dst dir) [otherwise some timesteps will be lost
        because one frame can be repeated after alignment]
        '''
        with open(os.path.join(self.rgb_output_directory, 'timesteps.pkl'), 'wb') as f:
            pickle.dump([os.path.basename(fp) for fp in dst_filepaths], f)
        with open(os.path.join(self.rgb_output_directory, 'timesteps.txt'), 'w') as f:
            f.write(str(["{}: {}".format(i, fp)
                         for i, fp in enumerate(dst_filepaths)]))

    def align_timesteps(self, reset_T0=False):
        """
        Align timesteps. Refer to match_timesteps() in this module for details.

        Parameters
        ----------
        reset_T0 : bool, optional
            If True delay of the inital frame will be removed from timestamps
        """
        indexes = tools.match_timesteps(*self._TPA_RGB_timestamps)
        #TPA
        for i in range(len(self.TPA.ids)):
            self.TPA.arrays[i] = self.TPA.arrays[i][indexes[i]]
            timestamps = np.array(self.TPA.timestamps[i])[indexes[i]]
            self.TPA.timestamps[i] = list(timestamps)
        #RGB
        i += 1
        self.RGB.timestamps = list(np.array(self.RGB.timestamps)[indexes[i]])
        self.RGB.filepaths = list(np.array(self.RGB.filepaths)[indexes[i]])
        #update timestamps
        self._update_TPA_RGB_timestamps()
        if reset_T0:
            sample_T0_min = np.min([ts[0] for ts in self._TPA_RGB_timestamps])
            timestamps = [list(np.array(ts)-sample_T0_min)
                          for ts in self._TPA_RGB_timestamps]
            self.TPA.timestamps = timestamps[:-1]
            self.RGB.timestamps = timestamps[-1]
            self._update_TPA_RGB_timestamps()
        return True


def _unpack_calib_pkl(fp: str) -> list:
    """
    Return content of calibration .pkl used in the project. 
    This function serves as a guide to formatting your own calibration matrix.
    Guide: .pkl should contain a dictionary of calib. info, e.g.:
    {'mtx': mtx, 'dist': dist, 'width':width, 'height': height}
    other keys are ignored
    from
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([...])

    Parameters
    ----------
    fp : str
        Filepath to calibration .pkl

    Returns 
    ------
    mtx 
        Calibration mtx.
    dist
        Calibraiton dist.
    widht
        Calibration width.
    height
        Calibration height.
    unparsed
        The rest of the original dictionary.
    """
    with open(fp, 'rb') as f:
        result = pickle.load(f)
    mtx = result['mtx']
    result.pop('mtx', None)
    dist = result['dist']
    result.pop('dist', None)
    width = result['width']
    result.pop('width', None)
    height = result['height']
    result.pop('height', None)
    unparsed_keys = result
    return mtx, dist, width, height, unparsed_keys


class _Undistorter():
    def __init__(self, mtx, dist, width, height):
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (width, height), 1, (width, height))
        self.mtx = mtx
        self.dist = dist
        self.newcameramtx, self.roi = newcameramtx, roi

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)


class TPA_RGB_Preparer(_Preparer):
    """
    Prepare files by processing raw samples (frame alignment) and generating a label file to be 
    filled by user needed for dataset generation. 
    - unproceesed sequences → aligned sequences and labels file (to be filled by user before making dataset)
    - filtering out samples that miss views (incomplete sequences)
    - aligning sequences
    - set HTPA32x32d.tools.SYNCHRONIZATION_MAX_ERROR in [s] that you're willing to tollerate
    Call generate_config_template() method to generate required config files.

    Input: 
    *ID*.TXT
    {config_making}.json
    tpa.nfo
    labels.json

    Output:
    *ID*.TXT
    tpa.nfo

    Arguments
    ---------
    configured : bool
        True if TPA_Preparer is ready to use prepare() method. Configure by calling config()
    Methods 
    -------
    generate_config_template()
        Generate config file template (json) to be filled by user 
        and passed to config()
    config()
        Configure #TODO FINISH DOCS
    
    """

    def __init__(self, reset_log=True):
        _Preparer.__init__(self, reset_log)

    def config(self, json_filepath):
        self._config(json_filepath)
        if (self.tpas_extension.lower() != 'txt'):
            msg = "[ERROR] Only .txt supported!"
            raise Exception(msg)
        lowercase_ids = [id.lower() for id in self.view_IDs]
        if 'rgb' in lowercase_ids:
            msg = "[ERROR] RGB in view_IDs skipped, RGB is handled by default, no need to fill it in view_IDs. Remove it from view_IDs."
            self._log(msg)
            raise Exception(msg)
        if self.undistort and (not os.path.exists(self.calib_fp)):
            msg = "[ERROR] {} doesn't exist while UNDISTORT is True".format(
                self.calib_fp)
            self._log(msg)
            raise Exception(msg)
        return True

    def prepare(self):
        if not self.configured:
            msg = "Configure with config() first"
            self._log(msg)
            raise Exception(msg)
        if not (self.raw_input_dir):
            msg = "Destination directory filepath not specified"
            self._log(msg)
            raise ValueError(msg)
        tools.ensure_path_exists(self.raw_input_dir)
        glob_patterns = [os.path.join(
            self.raw_input_dir, "*ID"+id+"."+self.tpas_extension) for id in self.view_IDs]
        files = []
        for pattern in glob_patterns:
            files.extend(glob.glob(pattern))
        prefixes = [_TPA_get_file_prefix(f) for f in files]
        prefixes2process = list(set(prefixes))
        prefixes2process_number0 = len(prefixes2process)
        # filter out samples that miss views
        prefixes2process = self._remove_missing_views(
            prefixes, prefixes2process)
        prefixes2process = self._remove_missing_rgbs(
            prefixes2process, self.raw_input_dir)
        prefixes2process_number = len(set(prefixes2process))
        prefixes_ignored = prefixes2process_number0 - prefixes2process_number
        self._log("[INFO] {} prefixes ignored out of initial {}".format(
            prefixes_ignored, prefixes2process_number0))
        self._log('"UNDISTORT" set to {}'.format(self.undistort))
        self._log("Reading, aligning and removing T0 from samples...")
        QUIT = False
        if self.undistort:
            mtx, dist, width, height, _ = _unpack_calib_pkl(self.calib_fp)
            self._undistorter = _Undistorter(mtx, dist, width, height)
        negative_samples_dict = {}
        for prefix in prefixes2process:
            raw_fp_prefix = os.path.join(self.raw_input_dir, prefix)
            self._log("Processing {}...".format(raw_fp_prefix))
            processed_fp_prefix = os.path.join(
                self.processed_destination_dir, prefix)
            tpa_fps = [raw_fp_prefix + "ID" + view_id + "." +
                       self.tpas_extension for view_id in self.view_IDs]
            processed_fps = [processed_fp_prefix + "ID" + view_id +
                             "." + self.tpas_extension for view_id in self.view_IDs]
            rgb_dir = os.path.join(self.raw_input_dir, prefix + "ID" + "RGB")
            processed_rgb_dir = os.path.join(
                self.processed_destination_dir, prefix + "ID" + "RGB")
            raw_sample = TPA_RGB_Sample_from_filepaths(tpa_fps, rgb_dir)
            header = raw_sample.get_header()
            processed_sample = TPA_RGB_Sample_from_data(raw_sample.TPA.arrays, raw_sample.TPA.timestamps, raw_sample.TPA.ids,
                                                        rgb_dir, tpa_output_filepaths=processed_fps, rgb_output_directory=processed_rgb_dir, header=header)
            processed_sample.align_timesteps(reset_T0=True)
            if not processed_sample.test_synchronization(max_error=SYNCHRONIZATION_MAX_ERROR):
                QUIT = True
                self._log("[ERROR] {} did not pass synchronization test (max error {} s exceeded)!".format(
                    prefix, SYNCHRONIZATION_MAX_ERROR))
                continue    
            if header.split(",")[-1] == "neg":
                negative_samples_dict[prefix] = int(-1)
            processed_sample.write()
            if self.visualize:
                processed_sample.write_gif()
            if self.undistort:
                img_fps = glob.glob(os.path.join(
                    processed_rgb_dir, "*." + tools.HTPA_UDP_MODULE_WEBCAM_IMG_EXT))
                for img_fp in img_fps:
                    img = cv2.imread(img_fp)
                    cv2.imwrite(img_fp, self._undistorter.undistort(img))
            self._log("Processed {}.".format(raw_fp_prefix))
        assert not QUIT
        self._write_nfo()
        self._write_labels_file(prefixes2process, negative_samples_dict)
        self._write_make_file()
        self._log("Writing nfo, labels and json files...")
        self._log("OK")


def _get_subject_from_header(header):
    '''
    Gets subject from header formated as: "subject_name, (...)"
    #TODO DOCS
    '''
    return header.split(",")[0]


def _get_label_from_json_file(prefix, json_filepath):
    with open(json_filepath) as f:
        data = json.load(f)
    return data[prefix]


def _get_class_from_json_file(prefix, json_filepath):
    t = _get_label_from_json_file(prefix, json_filepath)
    if (t > 0):
        return 1
    else:
        return 0


class TPA_RGB_Dataset_Maker(_Dataset_Maker):
    '''
    CALL A TPA_RGBPreparer FIRST ; #TODO FINISH DOCS
    ONLY TXT SUPPORTED
    '''

    def __init__(self, reset_log=True):
        _Dataset_Maker.__init__(self, reset_log)

    def config(self, json_filepath):
        self._config(json_filepath)
        return True

    def make(self):
        if not self.configured:
            msg = "Configure with config() first"
            self._log(msg)
            raise Exception(msg)
        if not (self.dataset_destination_dir):
            msg = "Destination directory filepath not specified"
            self._log(msg)
            raise ValueError(msg)
        tools.ensure_path_exists(self.dataset_destination_dir)
        glob_patterns = [os.path.join(
            self.processed_input_dir, "*ID"+id+"."+self.tpas_extension) for id in self.view_IDs]
        files = []
        for pattern in glob_patterns:
            files.extend(glob.glob(pattern))
        prefixes = [_TPA_get_file_prefix(f) for f in files]
        # processed samples are assumed to be aligned
        prefixes2make = list(set(prefixes))
        prefixes2make_number0 = len(prefixes2make)
        # filter out samples that miss views or RGBs
        prefixes2make = self._remove_missing_views(prefixes, prefixes2make)
        prefixes2make = self._remove_missing_rgbs(
            prefixes2make, self.processed_input_dir)
        # filter out samples that miss a label
        self._labels = self._read_labels_file(self.labels_filepath)
        prefixes2make_copy = prefixes2make.copy()
        for prefix in prefixes2make:
            if (prefix not in self._labels):
                prefixes2make_copy.remove(prefix)
                self._log('[WARNING] Ignoring prefix {} because it misses a label'.format(
                    prefix))
        prefixes2make = prefixes2make_copy.copy()
        for prefix in prefixes2make:
            if not self._labels[prefix]:
                prefixes2make_copy.remove(prefix)
                self._log('[WARNING] Ignoring prefix {} because it misses a label'.format(
                    prefix))
        prefixes2make = prefixes2make_copy.copy()
        for prefix in prefixes2make:
            if not (type(self._labels[prefix]) == int):
                prefixes2make_copy.remove(prefix)
                self._log('[WARNING] Ignoring prefix {} because the label is incorrect (it is not an integer)'.format(
                    prefix))
        prefixes2make = prefixes2make_copy.copy()
        # process the files
        fps2copy = []
        fps2output = []
        dirs2copy = []
        dirs2output = []
        label_txt_dict = {}
        for prefix in prefixes2make:
            subject_name = _get_subject_from_header(tools.read_txt_header(os.path.join(
                self.processed_input_dir, prefix + "ID" + self.view_IDs[0] + "." + self.tpas_extension)))
            subject_name = re.sub('[^\w\-_\. ]', '_', subject_name)
            # 1 > pos or 0 > neg
            label_class = 1 if (self._labels[prefix] > 0) else 0
            fp_prefix = os.path.join(self.processed_input_dir, prefix)
            fp_o_prefix = os.path.join(
                self.dataset_destination_dir, subject_name, str(label_class), prefix)
            tools.ensure_parent_exists(fp_o_prefix)
            fps = []
            fps_o = []
            for view_id in self.view_IDs:
                fp = fp_prefix + "ID" + view_id + "." + self.tpas_extension
                fp_o = fp_o_prefix + "ID" + view_id + "." + self.tpas_extension
                fps.append(fp)
                fps_o.append(fp_o)
            fp_rgb = fp_prefix + "ID" + "RGB"
            fp_o_rgb = fp_o_prefix + "ID" + "RGB"
            dirs2copy.append(fp_rgb)
            dirs2output.append(fp_o_rgb)
            fps2copy.append(fps)
            fps2output.append(fps_o)
            label_txt_dict[os.path.join(
                fp_o_rgb, "label.txt")] = self._labels[prefix]
        prefixes2make_number = len(set(prefixes2make))
        prefixes_ignored = prefixes2make_number0 - prefixes2make_number
        self._log("[INFO] {} prefixes ignored out of initial {}".format(
            prefixes_ignored, prefixes2make_number0))
        if (prefixes_ignored == prefixes2make_number0):
            self._log("[WARNING] All files ignored, the dataset is empty.")
            self._log("FAILED")
            return False
        self._log("[INFO] Making dataset...")
        self._log("[INFO] Copying files...")
        for src_tuple, dst_tuple in zip(fps2copy, fps2output):
            for src, dst in zip(src_tuple, dst_tuple):
                shutil.copy2(src, dst)
                old_header = tools.read_txt_header(src)
                tools.modify_txt_header(dst, old_header+",label{}".format(self._labels[_TPA_get_file_prefix(src)]))
        for src, dst in zip(dirs2copy, dirs2output):
            try:
                shutil.copytree(src, dst)
            except FileExistsError:
                shutil.rmtree(dst)
                shutil.copytree(src, dst)
        for fp in label_txt_dict.keys():
            with open(fp, "w") as f:
                f.write(str(label_txt_dict[fp]))
        self._log("Writing nfo, labels and json files...")
        self._write_nfo()
        self._copy_labels_file()
        self._log("OK")
        return True


def _pad_repeat_frames(array, first_n, last_n):
    def _pad_repeat_first_frame(array, n):
        padding = np.repeat(array[0].copy()[None, :], n, axis=0)
        return np.concatenate([padding, array], axis=0)

    def _pad_repeat_last_frame(array, n):
        padding = np.repeat(array[-1].copy()[None, :], n, axis=0)
        return np.concatenate([array, padding], axis=0)
    return _pad_repeat_first_frame(_pad_repeat_last_frame(array, last_n), first_n)


def _crop_and_repeat_ts(ts, start, end, first_n, last_n):
    array = np.array(ts)[start:end]

    def _pad_repeat_first_frame(array, n):
        padding = np.repeat(array[0].copy(), n, axis=0)
        return np.concatenate([padding, array], axis=0)

    def _pad_repeat_last_frame(array, n):
        padding = np.repeat(array[-1].copy(), n, axis=0)
        return np.concatenate([array, padding], axis=0)
    return _pad_repeat_first_frame(_pad_repeat_last_frame(array, last_n), first_n)

def _avg_ts(timestamp_list):
    return np.sum(timestamp_list, axis=0)/len(timestamp_list)

def convert_TXT2NPZ_TPA_RGB_Dataset(dataset_dir: str, frames: int, frame_shift: int = 0, output_dir: str = None, crop_to_center=True, size=None):
    '''
    Convert TXT dataset made with TPA_RGB_Dataset_Maker to NPZ,
    crop the recordings from array[:] to array[label-frames+frame_shift:label+frame_shift]
    size = (height, width) in pixels
    '''
    assert (frame_shift >= 0)
    assert (frames >= 0)
    assert (frames-frame_shift > 0)
    if not output_dir:
        output_dir = dataset_dir + "_npz_f{}_fs{}".format(frames, frame_shift)
    assert os.path.exists(dataset_dir)
    if size:
        assert (len(size) == 2)
        _h, _w =  size
        assert int(_h), int(_w)
    with open(os.path.join(dataset_dir, TPA_NFO_FN)) as f:
        data = json.load(f)
    view_IDs = data["view_IDs"]
    tpas_extension = data["tpas_extension"]
    neg_fps = glob.glob(os.path.join(dataset_dir, "*", "0", "*ID*"))
    pos_fps = glob.glob(os.path.join(dataset_dir, "*", "1", "*ID*"))
    files = neg_fps + pos_fps
    prefixes = list(set([_TPA_get_file_prefix(f) for f in files]))
    cnt = collections.Counter([_TPA_get_file_prefix(f) for f in files])
    fail = False
    for prefix in prefixes:
        if cnt[prefix] > (len(view_IDs) + 1):
            print("Prefix {} reused many times in the dataset!".format(prefix))
            fail = True
    if fail:
        return False
    for prefix in prefixes:
        print("Processing {}...".format(prefix))
        for f in files:
            if prefix in f:
                sample_dir = os.path.split(f)[0]
                continue
        sample_fp_prefix = os.path.join(sample_dir, prefix)
        sample_TPA_fps = [sample_fp_prefix+"ID"+view_id +
                          "."+tpas_extension for view_id in view_IDs]
        sample_RGB_fps = sample_fp_prefix+"ID"+"RGB"
        sample = TPA_RGB_Sample_from_filepaths(sample_TPA_fps, sample_RGB_fps)
        if (sample.label > 0):
            sample_class = DATASET_POSITIVE_ONE_HOT
            tpa_arrays = dict()
            tpa_timestamps = dict()
            old_label = sample.label
            new_label = frames-frame_shift
            old_length = len(sample.RGB.timestamps)
            if (old_label - frames + frame_shift > 0):
                start = old_label - frames + frame_shift
                pad_first = 0
            else:
                start = 0
                pad_first = -(old_label - frames + frame_shift)

            if (old_length - old_label - frame_shift > 0):
                end = old_label + frame_shift
                pad_last = 0
            else:
                end = old_length
                pad_last = -(old_length - old_label - frame_shift)
            for view_id, tpa_array, tpa_ts in zip(sample.TPA.ids, sample.TPA.arrays, sample.TPA.timestamps):
                tpa_arrays[view_id] = _pad_repeat_frames(
                    tpa_array[start:end], pad_first, pad_last).astype(np.half)
                tpa_timestamps[view_id] = _crop_and_repeat_ts(
                    tpa_ts, start, end, pad_first, pad_last)
            rgb_array = [cv2.imread(fp) for fp in sample.RGB.filepaths]
            rgb_array = tools.crop_center(np.array(_pad_repeat_frames(rgb_array[start:end], pad_first, pad_last)).astype(np.uint8))
            if size:
                rgb_array = np.array([cv2.resize(img, (int(size[1]), int(size[0]))) for img in rgb_array], dtype=np.uint8)
            rgb_timestamps = _crop_and_repeat_ts(
                sample.RGB.timestamps, start, end, pad_first, pad_last)
            tpa_avg_timestamps = _avg_ts(list(tpa_timestamps.values()))
            tpa_rgb_avg_timestamps = _avg_ts(list(tpa_timestamps.values()) + [rgb_timestamps])
            optional_kwargs = {"pad_first": pad_first,
                               "pad_last": pad_last, "repeating_frames": True}
            output_fp = os.path.join(output_dir, os.path.relpath(
                sample_dir, dataset_dir), "{}.npz".format(prefix))
            tools.ensure_parent_exists(output_fp)
            np.savez_compressed(output_fp, one_hot=sample_class, frames=frames, frame_shift=frame_shift, tpa_avg_timestamps=tpa_avg_timestamps, tpa_rgb_avg_timestamps=tpa_rgb_avg_timestamps,  **optional_kwargs, **{"array_ID{}".format(view_id): tpa_arrays[view_id] for view_id in sample.TPA.ids}, **{
                                "timestamps_ID{}".format(view_id): tpa_timestamps[view_id] for view_id in sample.TPA.ids}, **{'array_IDRGB': rgb_array, 'timestamps_IDRGB': rgb_timestamps})
        if (sample.label <= 0):
            sample_class = DATASET_NEGATIVE_ONE_HOT
            tpa_arrays = dict()
            tpa_timestamps = dict()
            old_length = len(sample.RGB.timestamps)
            if (old_length - frames > 0):
                start = 0
                pad_first = 0
            else:
                start = 0
                pad_first = -(old_length - frames)
            end = frames
            pad_last = 0
            for view_id, tpa_array, tpa_ts in zip(sample.TPA.ids, sample.TPA.arrays, sample.TPA.timestamps):
                tpa_arrays[view_id] = _pad_repeat_frames(
                    tpa_array[start:old_length], pad_first, pad_last).astype(np.half)
                tpa_timestamps[view_id] = _crop_and_repeat_ts(
                    tpa_ts, start, old_length, pad_first, pad_last)
            rgb_array = [cv2.imread(fp) for fp in sample.RGB.filepaths]
            rgb_array = tools.crop_center(np.array(_pad_repeat_frames(rgb_array[start:old_length], pad_first, pad_last)).astype(np.uint8))
            if size:
                rgb_array = np.array([cv2.resize(img, (int(size[1]), int(size[0]))) for img in rgb_array], dtype=np.uint8)
            rgb_timestamps = _crop_and_repeat_ts(
                sample.RGB.timestamps, start, old_length, pad_first, pad_last)
            # sliding window
            patches_n = old_length//frames
            rgb_patches = [rgb_array[i*frames:(i+1)*frames]
                           for i in range(patches_n)]
            tpa_patches = [{view_id: tpa_arrays[view_id][i*frames:(
                i+1)*frames] for view_id in sample.TPA.ids} for i in range(patches_n)]
            rgb_timestamps_patches = [_crop_and_repeat_ts(
                rgb_timestamps, i*frames, (i+1)*frames, 0, 0) for i in range(patches_n)]
            tpa_timestamps_patches = [{view_id: _crop_and_repeat_ts(
                tpa_timestamps[view_id], i*frames, (i+1)*frames, 0, 0) for view_id in sample.TPA.ids} for i in range(patches_n)]
            for i, (tpa_patch, rgb_patch, tpa_ts, rgb_ts) in enumerate(zip(tpa_patches, rgb_patches, tpa_timestamps_patches, rgb_timestamps_patches)):  
                tpa_avg_timestamps = _avg_ts(list(tpa_ts.values()))
                tpa_rgb_avg_timestamps = _avg_ts(list(tpa_ts.values()) + [rgb_ts])
                output_fp = os.path.join(output_dir, os.path.relpath(
                    sample_dir, dataset_dir), "patch{}_{}.npz".format(i, prefix))
                tools.ensure_parent_exists(output_fp)
                optional_kwargs = {"pad_first": pad_first if (i == 0) else int(0), "pad_last": int(0), "repeating_frames": True if (i == 0) else False}
                np.savez_compressed(output_fp, one_hot=sample_class, frames=frames, frame_shift=0, tpa_avg_timestamps=tpa_avg_timestamps, tpa_rgb_avg_timestamps=tpa_rgb_avg_timestamps, **optional_kwargs, **{"array_ID{}".format(view_id): tpa_patch[view_id] for view_id in sample.TPA.ids}, **{
                                    "timestamps_ID{}".format(view_id): tpa_ts[view_id] for view_id in sample.TPA.ids}, **{'array_IDRGB': rgb_patch, 'timestamps_IDRGB': rgb_ts})
    rgb_shape = None
    if size:
        rgb_shape = list(size) + [3]
    else:
        rgb_shape = rgb_array.shape
    data = {"repeating_frames": True, "frame_shift" : frame_shift, "frames" : frames, "view_IDs" : view_IDs, "rgb_shape" : rgb_shape}
    with open(os.path.join(output_dir, "samples.nfo"), 'w') as f:
        json.dump(data, f)

