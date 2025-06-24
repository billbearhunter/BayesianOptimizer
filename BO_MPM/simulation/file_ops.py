import os
import taichi as ti
import csv
import ctypes
import numpy as np

@ti.data_oriented
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
            writer.writerow([f"{v:.16e}" for v in data])


        with open(dat_path, 'a') as dat_file:
            dat_file.write(' '.join([f"{v:.16e}" for v in data]) + '\n')

        return csv_path, dat_path
    

@ti.data_oriented
class FileOperations:
    def __init__(self):
        self.py_saved_iteration = 0
        self.py_filename = ''
        self.py_save_count = 1

        self.py_root_dir_path = 'data'
        self.py_file_processing = ''
    
    
    def saveFile(self, agTaichiMPM, output_dir):
        self.py_save_count  = agTaichiMPM.py_num_saved_frames
        # print("output_dir: ", output_dir)
        saveStateFilePath = os.path.join(output_dir, 'config_' + str(self.py_save_count).zfill(2) + ".dat")
        saveStateIntermediateFilePath = os.path.join(output_dir, 'config_' + str(self.py_save_count).zfill(2) + "_phi" + ".dat")
        outObjFilePath = os.path.join(output_dir, 'config_' + str(self.py_save_count).zfill(2) + ".obj")
        particleSkinnerApp = 'ParticleSkinner3DTaichi/ParticleSkinner3DTaichi.py'
        marching_cube_path = 'ParticleSkinner3DTaichi/cpp_marching_cubes/build/cpp_marching_cubes'
 
        for filepath in [saveStateFilePath, saveStateIntermediateFilePath, outObjFilePath]:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        # marching_cube_path = os.path.join('ParticleSkinner3DTaichi/cpp_marching_cubes/build/cpp_marching_cubes')
        # agTaichiMPM.particleSkinnerApp = os.path.join('ParticleSkinner3DTaichi/ParticleSkinner3DTaichi.py')

        print('[AGTaichiMPM] saving state to ' + saveStateFilePath)
        f = open(saveStateFilePath, 'wb')
        particle_is_inner_of_box_id = np.where(agTaichiMPM.ti_particle_is_inner_of_box.to_numpy()[0:agTaichiMPM.ti_particle_count[None]].astype(np.int32) == 1)
        f.write(ctypes.c_int32(agTaichiMPM.ti_particle_count[None] -  particle_is_inner_of_box_id[0].size))
        #output x
        p_x = agTaichiMPM.ti_particle_x.to_numpy()[0:agTaichiMPM.ti_particle_count[None]].astype(np.float32)
        np.delete(p_x, particle_is_inner_of_box_id,axis=0).flatten().tofile(f)
        #output radius
        np.delete((np.ones(agTaichiMPM.ti_particle_count[None], np.float32) * agTaichiMPM.py_particle_hl).astype(np.float32), particle_is_inner_of_box_id,axis=0).flatten().tofile(f)
        #output velocity
        np.delete(agTaichiMPM.ti_particle_v.to_numpy()[0:agTaichiMPM.ti_particle_count[None]].astype(np.float32), particle_is_inner_of_box_id,axis=0).flatten().tofile(f)
        #output id
        np.delete(np.ones(agTaichiMPM.ti_particle_count[None], ctypes.c_int32), particle_is_inner_of_box_id,axis=0).flatten().tofile(f)
        f.close()