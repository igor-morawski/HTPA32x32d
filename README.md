# HTPA32x32d 

This repository contains tools to work with thermopile sensor array Heimann HTPA 32x32d and UDP communication module (included in the starting kit) in multi-view setup.

Thermopile sensor array is an extremely low-resolution far-infrared imaging sensor (camera). This sensor can preserve the privacy of people captured, so I use this sensor in my on-going research on using thermopile sensor arrays in monitoring systems capable of early event detection/risk assessment/accident prevention.

This repo is developed for HTPA 32x32d but can be easily modifed to generalize to any device resolution.
## Learn more
You can learn about the sensor more by visiting [Heimann Sensor's website](https://www.heimannsensor.com/products_imaging.php), or you can learn more about me and my research [on my website](https://igor-morawski.github.io/).

Feel free to [contact me](https://www.linkedin.com/in/igor-morawski/) if you have any questions about the sensor or the research.

## tools.py
A collection of useful functions and data structures for working with data captured by Heimann HTPA32x32d and other thermopile sensor arrays. 

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

