
  Class  | Input | Output | Remarks
------------- | ------------- | ------------- | -------------
TPA_Preparer  | `*ID*.TXT` (unprocessed) <br> `{config_procesing}.json`| `*ID*.TXT` <br> `tpa.nfo` <br> `labels.json` (to be filled by user) <br> `{config_making}.json` (to be filled by user) | unproceesed sequences → aligned sequences and labels file (to be filled by user before making dataset) <br>  filtering out samples that miss views (incomplete sequences) <br> - aligning sequences <br> - set HTPA32x32d.tools.SYNCHRONIZATION_MAX_ERROR in [s] that you're willing to tollerate
TPA_Dataset_Maker  | `*ID*.TXT` (processed: aligned) <br> `{config_making}.json` <br> `tpa.nfo` <br> `labels.json` | `*ID*.TXT` <br> `tpa.nfo` | aligned sequences and labels file (filled by user) → aligned sequences and labels file <br>  - filtering out samples that miss views (incomplete sequences) <br> - filtering out samples that miss a label
TPA_Sample_from_filepaths | list of fps | N/A | N/A 
TPA_Sample_from_data | list of arrays, list of timestamps and list of ids | N/A |  N/A 

Call *generate_config_template()* method to generate required config files. 

#### Pipeline
Example:

first make a conifg file (generate a template if you don't have one yet)
```
import HTPA32x32d
HTPA32x32d.tools.VERBOSE = True
import os
raw_dir = "raw" # this is your directory that contains raw .TXT files from HTPA32x32d, all named YYYYMMDD_HHmm_ID{id}.TXT
preparer = HTPA32x32d.tools.TPA_Preparer()
preparer.generate_config_template(os.path.join(raw_dir, "config.json"))
```
Now fill your config.json, e.g.:
```
{
    "raw_input_dir": "path_to_your_raw_dir",
    "processed_destination_dir": "path_to_your_processed_dir",
    "view_IDs": ["121", "122", "123"],
    "tpas_extension": "TXT",
    "MAKE": 0,
    "PREPARE": 1,
    "VISUALIZE": 1
}
```
Now process your raw files to align them and generate labels.json file; samples that are incomplete (missing view) will be ignored
```
import HTPA32x32d
HTPA32x32d.tools.VERBOSE = True
import os
raw_dir = "raw" # this is your directory that contains raw .TXT files from HTPA32x32d, all named YYYYMMDD_HHmm_ID{id}.TXT
preparer = HTPA32x32d.tools.TPA_Preparer()
preparer.config(os.path.join(raw_dir, "config.json"))
HTPA32x32d.tools.SYNCHRONIZATION_MAX_ERROR = 0.15
preparer.prepare()
```
Now fill in all the labels that you want to fill in; you can use scripts in the "scripts" directory to convert gifs to mp4 (and enlarge them and label a timestep at each frame) to help you identify your labels.  Samples with no labels or that are incomplete (missing view) will be ignored.
fill in your dataset destination in generated `make_config.json` file 
```
import HTPA32x32d
HTPA32x32d.tools.VERBOSE = True
import os
processed_dir = "processed" # this is your directory that contains raw .TXT files from HTPA32x32d, all named YYYYMMDD_HHmm_ID{id}.TXT
maker = HTPA32x32d.tools.TPA_Dataset_Maker()
maker.config(os.path.join(processed_dir, "make_config.json"))
maker.make()
```
