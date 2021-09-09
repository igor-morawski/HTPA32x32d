Feel free to [contact me](https://www.linkedin.com/in/igor-morawski/) if you have any questions about the sensor or the research.

If you need to use this module to work with Heimann HTPA32x32d, I suggest forking the repo because maintaining backwards compatibility (as well as documentation) is not my priority at the moment. 


```BibTeX

@INPROCEEDINGS{9506024,
  author={Morawski, Igor and Lie, Wen-Nung and Chiang, Jui-Chiu},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)}, 
  title={Action Prediction Using Extremely Low-Resolution Thermopile Sensor Array For Elderly Monitoring}, 
  year={2021},
  volume={},
  number={},
  pages={984-988},
  doi={10.1109/ICIP42928.2021.9506024}}


@inproceedings{morawski2020two,
  title={Two-stream deep learning architecture for action recognition by using extremely low-resolution infrared thermopile arrays},
  author={Morawski, Igor and Lie, Wen-Nung},
  booktitle={International Workshop on Advanced Imaging Technology (IWAIT) 2020},
  volume={11515},
  pages={115150Y},
  year={2020},
  organization={International Society for Optics and Photonics}
}


@misc{im2020HTPA32x32d,
  author =       {Igor Morawski},
  title =        {HTPA32x32d tools},
  howpublished = {\url{https://github.com/igor-morawski/HTPA32x32d/}},
  year =         {2020}
}
```

# HTPA32x32d 

This repository contains tools to work with thermopile sensor array Heimann HTPA 32x32d and UDP communication module (included in the starting kit) in multi-view setup. Because it is still uner development for my research, there might be **no backward compability** between the commits.

Thermopile sensor array is an extremely low-resolution far-infrared imaging sensor (camera). This sensor can preserve the privacy of people captured, so I use this sensor in my on-going research on using thermopile sensor arrays in monitoring systems capable of early event detection/risk assessment/accident prevention.

This repo is developed for HTPA 32x32d but can be easily modifed to generalize to any device resolution.
## Learn more
You can learn about the sensor more by visiting [Heimann Sensor's website](https://www.heimannsensor.com/products_imaging.php), or you can learn more about me and my research [on my website](https://igor-morawski.github.io/). You can also contact me on [LinkedIn](https://www.linkedin.com/in/igor-morawski/).


































# OLD README: 
## tools.py
A collection of useful functions and data structures for working with data captured by Heimann HTPA32x32d and other thermopile sensor arrays. 

### Data types supported
* Call `SUPPORTED_EXTENSIONS` to see the list of currently supported types.
  * txt ⟵ currently the only extension that can copy file headers (the first line in Heimanns HTPA recordings)
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

### Samples - data structures:
Samples now support visualization by using write_gif() \[given that the data is aligned\].
* `TPA_Sample_from_data `
* `TPA_Sample_from_filepaths`
* `TPA_RGB_Sample_from_data` 
* `TPA_RGB_Sample_from_filepaths` 

### Dataset making
See [examples/dataset_making](https://github.com/igor-morawski/HTPA32x32d/blob/master/examples/dataset_making/README.md) and [examples/dataset_making/README.md](https://github.com/igor-morawski/HTPA32x32d/blob/master/examples/dataset_making/README.md).

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

