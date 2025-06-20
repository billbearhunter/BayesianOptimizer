import os
import csv
import ctypes
import numpy as np

class FileOperations:
    def __init__(self):
        self.py_saved_iteration = 0
        self.py_filename = ''
        self.py_save_count = 1
        self.py_root_dir_path = 'data'
        self.py_file_processing = ''
    
    def save_data(self, data, output_dir, file_prefix, frame_count):
        csv_path = os.path.join(output_dir, f"{file_prefix}.csv")
        dat_path = os.path.join(output_dir, f"{file_prefix}.dat")
        

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        
        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data)
            
        
        with open(dat_path, 'a') as dat_file:
            dat_file.write(' '.join(map(str, data)) + '\n')
            
        return csv_path, dat_path