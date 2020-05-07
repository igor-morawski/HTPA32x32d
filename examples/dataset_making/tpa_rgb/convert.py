import HTPA32x32d
HTPA32x32d.dataset.VERBOSE = True
import os
dataset_dir = "dataset" # this is your directory that contains raw .TXT files from HTPA32x32d, all named YYYYMMDD_HHmm_ID{id}.TXT
HTPA32x32d.dataset.convert_TXT2NPZ_TPA_RGB_Dataset(dataset_dir, frames=100, frame_shift=20)
