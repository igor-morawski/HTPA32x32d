Feel free to [contact me](https://www.linkedin.com/in/igor-morawski/) if you have any questions about the sensor or the research.

# HTPA32x32d 

This repository contains tools to work with thermopile sensor array Heimann HTPA 32x32d and UDP communication module (included in the starting kit) in multi-view setup. Because it is still uner development for my research, there might be **no backward compability** between the commits.

Thermopile sensor array is an extremely low-resolution far-infrared imaging sensor (camera). This sensor can preserve the privacy of people captured, so I use this sensor in my on-going research on using thermopile sensor arrays in monitoring systems capable of early event detection/risk assessment/accident prevention.

This repo is developed for HTPA 32x32d but can be easily modifed to generalize to any device resolution.
## Learn more
You can learn about the sensor more by visiting [Heimann Sensor's website](https://www.heimannsensor.com/products_imaging.php), or you can learn more about me and my research [on my website](https://igor-morawski.github.io/). You can also contact me on [LinkedIn](https://www.linkedin.com/in/igor-morawski/).

## tools.py
A collection of useful functions and data structures for working with data captured by Heimann HTPA32x32d and other thermopile sensor arrays. 

### Data types supported
* Call `SUPPORTED_EXTENSIONS` to see the list of currently supported types.
  * txt
  * csv
  * pickle (.pickle, .pkl, .p)

### Reading and writing files
* `read_tpa_file` reads files with supported extensions (deduced from filename extension given as argument)

* `write_tpa_file`  writes files with supported extensions (deduced from filename extension given as argument)

### Visualization
* `apply_heatmap` applies opencv (cv2) heatmaps
* `np2pc` temperature numpy array to pseudocolored numpy array 
* `write_pc2gif` pseudocolored RGB sequence to animated gif, timing between frames is kept if duration is passed 
  * use `timestamps2frame_durations` if you need to convert timestamps to frame durations
  
### Aligning, resampling, cropping
* `match_timesteps` to get indexes of timestamps so that timestamp\[corresponding_index_list\] is aligned with other given timestamps
* `resampling`
* `crop_center` to keep only center portion of the sequence, e.g. 28x28 out of 32x32 pixels

### Dataset making

  Class  | Input | Output | Remarks
------------- | ------------- | ------------- | -------------
TPA_Preparer  | `*ID*.TXT` (unprocessed) <br> `{config_procesing}.json`| `*ID*.TXT` <br> `tpa.nfo` <br> `labels.json` (to be filled by user) <br> `{config_making}.json` (to be filled by user) | unproceesed sequences → aligned sequences and labels file (to be filled by user before making dataset) <br>  filtering out samples that miss views (incomplete sequences) <br> - aligning sequences 
TPA_Dataset_Maker  | `*ID*.TXT` (processed: aligned) <br> `{config_making}.json` <br> `tpa.nfo` <br> `labels.json` | `*ID*.TXT` <br> `tpa.nfo` | aligned sequences and labels file (filled by user) → aligned sequences and labels file <br>  - filtering out samples that miss views (incomplete sequences) <br> - filtering out samples that miss a label

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
now fill your config.json, e.g.:
```
{
    "raw_input_dir": "path_to_your_raw_dir",
    "processed_destination_dir": "path_to_your_processed_dir",
    "view_IDs": ["121", "122", "123"],
    "tpas_extension": "TXT",
    "MAKE": 0,
    "PREPARE": 1
}
```
now process your raw files to align them and generate labels.json file; samples that are incomplete (missing view) will be ignored
```
import HTPA32x32d
HTPA32x32d.tools.VERBOSE = True
import os
raw_dir = "raw" # this is your directory that contains raw .TXT files from HTPA32x32d, all named YYYYMMDD_HHmm_ID{id}.TXT
preparer = HTPA32x32d.tools.TPA_Preparer()
preparer.config(os.path.join(raw_dir, "config.json"))
preparer.prepare()
```
now fill in all the labels that you want to fill in; samples with no labels or that are incomplete (missing view) will be ignored
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
## recorder.py
Python program that connects to Heimann HTPA sensors given their IP addresses (in settings file) and records data captured to TXT files. Supports recording mutliple sensors at the same time. This tool is supposed to help developing multi-view thermopile sensor array monitoring system. Number of the cameras it can connect to is unlimited. 


## converter.py
Python program that converts TXT files recorded by Heimann HTPA sensors and 

To get help run (depedning on the OS, you might need to change `python` to `python3`):
```
python converter.py --help
```

### Usage example
```
python converter.py --csv --gif --bmp --crop=26 --overwrite DIRECTORY_CONTAINING_FILES__OR__SINGLE_FILE_PATH
```
Equivalent to:
```
python converter.py -c -g -b --crop=26 --overwrite DIRECTORY_CONTAINING_FILES__OR__SINGLE_FILE_PATH
```

#### Explenation:

Converts all TXT files in a given directory or a given file:

`--csv` to CSV file,

`--gif` to GIF annimation preserving original frame durations,

`--bmp` extracts frames to a directory named after the filename, 

`--crop` in pixels, data frames are cropped to a patch of a given size in the center of the frame (note: CSV is never affected by this flag),

`--overwrite` overwrites the files if they already exists.

`--debug` debugs corrupted .TXT files.


## misc/photocap.py
Python program that connects to Heimann HTPA sensors given their IP addresses (in settings file) and captures data (single frames) to TXT files. Supports recording mutliple sensors at the same time. This tool is supposed to help developing multi-view thermopile sensor array monitoring system. Number of the cameras it can connect to is unlimited. 

## misc/img_converter.py
Python program that converts TXT files (single-frame files) recorded by Heimann HTPA sensors and calculates and saves histograms.

```BibTeX
@misc{im2020HTPA32x32d,
  author =       {Igor Morawski},
  title =        {HTPA32x32d tools},
  howpublished = {\url{https://github.com/igor-morawski/HTPA32x32d/}},
  year =         {2020}
}
```
