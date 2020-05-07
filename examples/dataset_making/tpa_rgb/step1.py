import HTPA32x32d
HTPA32x32d.dataset.VERBOSE = True
import os
raw_dir = "raw" # this is your directory that contains raw .TXT files from HTPA32x32d, all named YYYYMMDD_HHmm_ID{id}.TXT
preparer = HTPA32x32d.dataset.TPA_RGB_Preparer()
preparer.generate_config_template(os.path.join(raw_dir, "config.json"))
