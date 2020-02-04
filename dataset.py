"""
Utility to create and manage dataset of thermopile sensor arrays recordings in multi-view setup and deep learning. 
"""
import glob
import os
import tools
from tqdm import tqdm

# TODO testing

class Dataset_converter_txt2csv:
    def __init__(self, directory_path, dataset_name='dataset_csv', destination_path=None):
        """
        Creates a dataset of multi-view thermopile sensor array recording by converting from *.TXT files to *CSV files. 
        Globs recursively. Preseves directory structure.
        
        Parameters
        ----------
        directory_path : str
            Path to a directory containing all the TXT files. 

        destination_path : str, optional
            Path to a parent directory to write dataset to. By default inferred from directory_path and equal to directory_path's parent directory.
        """
        self._file_extension = 'TXT'
        self._new_file_extension = 'csv'
        self._text_files = glob.glob(os.path.join(
            directory_path, '**', '*.{}'.format(self._file_extension)))
        self._directory_path = directory_path
        if destination_path:
            self.destination_path = destination_path
        else:
            self.destination_path = os.path.dirname(directory_path)
        self.dataset_name = dataset_name

    def create(self):
        """
        Convert *.TXT and write *.CSV files preserving directory structure
        """
        dataset_path = os.path.join(self.destination_path, self.dataset_name)
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        for file_path in tqdm(self._text_files):
            new_relative_path = os.path.relpath(file_path, self._directory_path).replace(
                '.{}'.format(self._file_extension), '.{}'.format(self._new_file_extension))
            new_file_path = os.path.join(self.destination_path, self.dataset_name, new_relative_path)
            if not os.path.exists(os.path.dirname(new_file_path)):
                os.mkdir(os.path.dirname(new_file_path))
            data = tools.txt2np(file_path)
            tools.write_np2csv(new_file_path, *data)


if __name__ == '__main__':
    dummy_path = 'E:\\tmp\\try'
    dataset_txt2csv = Dataset_converter_txt2csv(dummy_path)
    dataset_txt2csv.create()
    self = dataset_txt2csv
