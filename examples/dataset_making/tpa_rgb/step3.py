import HTPA32x32d
HTPA32x32d.tools.VERBOSE = True
import os
processed_dir = "processed" # this is your directory that contains raw .TXT files from HTPA32x32d, all named YYYYMMDD_HHmm_ID{id}.TXT
maker = HTPA32x32d.tools.TPA_RGB_Dataset_Maker()
maker.config(os.path.join(processed_dir, "make_config.json"))
maker.make()
