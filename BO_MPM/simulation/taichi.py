import taichi as ti
import numpy as np
import gc
from .xmlParser import MPMXMLData
from .file_ops import FileOperations
from config.config import *

ti.init(arch=ti.gpu, offline_cache=True)

class MPMSimulator:
    def __init__(self, xml_config_path):
        self.xml_data = MPMXMLData(xml_config_path)
        self.file_ops = FileOperations()
        self.agtaichi_mpm = self._initialize_mpm()
        
    def _initialize_mpm(self):
        """Create and initialize MPM instance"""
        agtaichi_mpm = AGTaichiMPM(self.xml_data)
        agtaichi_mpm.changeSetUpData(self.xml_data)
        agtaichi_mpm.initialize()
        return agtaichi_mpm
    
    def configure_geometry(self, width, height):
        """Configure geometry parameters"""
        if not (MIN_WIDTH <= width <= MAX_WIDTH):
            raise ValueError(f"Width must be between {MIN_WIDTH} and {MAX_WIDTH}")
        if not (MIN_HEIGHT <= height <= MAX_HEIGHT):
            raise ValueError(f"Height must be between {MIN_HEIGHT} and {MAX_HEIGHT}")
            
        new_max_value = [width, height, 4.15]
        self.xml_data.cuboidData.max = new_max_value
        self.xml_data.staticBoxList[2].max[0] = width
        self.xml_data.staticBoxList[3].max[0] = width
        self.agtaichi_mpm.changeSetUpData(self.xml_data)
        
    def run_simulation(self, n, eta, sigma_y):
        """Run a single simulation and return displacement results"""
        # Validate parameters
        self._validate_params(n, eta, sigma_y)
        
        # Configure material parameters
        self.xml_data.integratorData.herschel_bulkley_power = n
        self.xml_data.integratorData.eta = eta
        self.xml_data.integratorData.yield_stress = sigma_y
        
        # Reset simulator
        self.agtaichi_mpm.changeSetUpData(self.xml_data)
        self.agtaichi_mpm.initialize()
        self.agtaichi_mpm.py_num_saved_frames = 0
        
        # Execute simulation
        return self._execute_simulation_loop()
    
    def _validate_params(self, n, eta, sigma_y):
        """Validate parameter ranges"""
        if not (MIN_N <= n <= MAX_N):
            raise ValueError(f"n must be between {MIN_N} and {MAX_N}")
        if not (MIN_ETA <= eta <= MAX_ETA):
            raise ValueError(f"eta must be between {MIN_ETA} and {MAX_ETA}")
        if not (MIN_SIGMA_Y <= sigma_y <= MAX_SIGMA_Y):
            raise ValueError(f"sigma_y must be between {MIN_SIGMA_Y} and {MAX_SIGMA_Y}")
    
    def _execute_simulation_loop(self):
        """Execute simulation main loop"""
        x_diffs = []
        x_0frame = 0
        
        while self.agtaichi_mpm.py_num_saved_frames <= self.agtaichi_mpm.py_max_frames:
            # Execute multiple time steps
            for _ in range(100):
                self.agtaichi_mpm.step()
                time = self.agtaichi_mpm.ti_iteration[None] * self.agtaichi_mpm.py_dt
                
                # Save frame data
                if time * self.agtaichi_mpm.py_fps >= self.agtaichi_mpm.py_num_saved_frames:
                    particle_data = self._get_particle_positions()
                    if self.agtaichi_mpm.py_num_saved_frames == 0:    
                        x_0frame = np.max(particle_data[:, 0])
                    elif self.agtaichi_mpm.py_num_saved_frames > 0:
                        x_diff = np.max(particle_data[:, 0]) - x_0frame
                        x_diffs.append(x_diff)
                    
                    self.agtaichi_mpm.py_num_saved_frames += 1
            
            # Check termination condition
            if self.agtaichi_mpm.py_num_saved_frames > self.agtaichi_mpm.py_max_frames:
                break
                
        gc.collect()
        return np.array(x_diffs)
    
    def _get_particle_positions(self):
        """Get particle position data (optimized memory usage)"""
        particle_is_inner = self.agtaichi_mpm.ti_particle_is_inner_of_box.to_numpy()[
            0:self.agtaichi_mpm.ti_particle_count[None]].astype(np.int32) == 1
        p_x = self.agtaichi_mpm.ti_particle_x.to_numpy()[
            0:self.agtaichi_mpm.ti_particle_count[None]].astype(np.float32)
        return p_x[~particle_is_inner]
    
    def cleanup(self):
        """Clean up resources"""
        self.agtaichi_mpm.cleanup()
        ti.reset()