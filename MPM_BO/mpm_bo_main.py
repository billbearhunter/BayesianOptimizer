import taichi as ti
import numpy as np
import xml.etree.ElementTree as ET
import sys
import csv
import xmlParser
import os
import ctypes
import psutil
import gc
import time
import pandas as pd

import torch
import time
from base_optimizer import BaseBayesianOptimizer
from multitask_optimizer import MultiTaskBayesianOptimizer
from xmlParser import MPMXMLData

ti.init(arch=ti.gpu)
# ti.init(arch=ti.cuda)
# process = psutil.Process(os.getpid())

# Define parameter ranges and generate their values
MIN_ETA = 0.001
MAX_ETA = 300.0
MIN_N = 0.3
MAX_N = 1.0
MIN_SIGMA_Y = 0.0
MAX_SIGMA_Y = 400.0

MIN_WIDTH = 2.0
MAX_WIDTH = 7.0
MIN_HEIGHT = 2.0
MAX_HEIGHT = 7.0

# BOUNDS_DESIGN = torch.tensor([
#     [-4.0,  -2.0,  0.3,  -1.70, -1.70],   # min logη, logσY, n, logw, logh
#     [ 3.0,   2.0,  1.0,  -1.15, -1.15],   # max
# ])

BOUNDS = [
    [MIN_N, MAX_N],
    [MIN_ETA, MAX_ETA],
    [MIN_SIGMA_Y, MAX_SIGMA_Y]
]


def objective_wrapper(x: np.ndarray) -> np.ndarray:
    """
    Objective function to be minimized.
    x: [n, eta, sigma_y]
    """
    n, eta, sigma_y = x
    width, height = 3.9, 6.8  
    print(f"Evaluating with n={n:.4f}, eta={eta:.4f}, sigma_y={sigma_y:.4f}")

    try:
        
        xmlData.integratorData.herschel_bulkley_power = n
        xmlData.integratorData.eta = eta
        xmlData.integratorData.yield_stress = sigma_y
        
        new_max_value = [width, height, 4.15]
        xmlData.cuboidData.max = new_max_value
        xmlData.staticBoxList[2].max[0] = width
        xmlData.staticBoxList[3].max[0] = width
        
        x_diffs = MPM_simulation(n, eta, sigma_y, xmlData)
        if x_diffs.size != 8:
            raise ValueError(f"Expected 8 outputs, got {x_diffs.size}")
        return x_diffs.astype(np.float64)
    except Exception as e:
        print(f"Error during simulation: {e}")
        return np.full(8, np.nan)

def get_data(data_path):

    data = pd.read_csv(data_path)
    y = data.iloc[:, 3:].values.astype(np.float64)
    X_hat = data.iloc[:, :3].values.astype(np.float64)

    X_hat = X_hat.astype(np.float64)
    y = y.astype(np.float64)

    return X_hat, y


# Define a function to extract particle data from a .dat file
def extract_particle_data_x(dat_file_path):
    with open(dat_file_path, 'rb') as file:
        num_points = int.from_bytes(file.read(4), 'little')
        positions = np.frombuffer(file.read(3 * num_points * 4), dtype=np.float32).reshape((num_points, 3))
    return positions[:, 0].max()


def parseXML(xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    return root

@ti.data_oriented
class AGTaichiMPM:
    
    def __init__(self, xmlData):      
        self.ti_hb_n = ti.field(float, ())
        self.ti_hb_eta = ti.field(float, ())
        self.ti_hb_sigmaY = ti.field(float, ())

        self.ti_num_boxes = ti.field(int, ())
        self.ti_num_boxes[None] = 4
        self.ti_static_box_min = ti.Vector.field(3, float, self.ti_num_boxes[None])
        self.ti_static_box_max = ti.Vector.field(3, float, self.ti_num_boxes[None])
        self.ti_static_box_type = ti.field(ti.i32, self.ti_num_boxes[None])

        self.py_fps = xmlData.integratorData.fps
        # material parameters
        self.py_kappa     = xmlData.integratorData.bulk_modulus
        self.py_mu        = xmlData.integratorData.shear_modulus
        print('py_kappa: ', self.py_kappa)
        print('py_mu: ', self.py_mu)

        # flip-pic alpha
        self.py_alpha = xmlData.integratorData.flip_pic_alpha

        # temporal/spatial resolution
        self.py_dt = xmlData.integratorData.dt
        self.py_dx = xmlData.gridData.cell_width
        self.py_invdx = 1.0 / self.py_dx

        # near earth gravity
        self.ti_g = ti.Vector(xmlData.nearEarthGravityData.g)
        print('ti_g: ', self.ti_g)

        # iteration count
        self.ti_iteration = ti.field(int, ())
        self.ti_iteration[None] = 0

        # max time
        self.py_max_frames = xmlData.integratorData.max_frames
        self.py_num_saved_frames = 0
        print('py_max_frames: ', self.py_max_frames)

        # configuring grid by using the specified grid center and cell width as is
        # min and max will be recomputed because the specified grid size may not agree with the specified cell width

        # compute grid center and tentative grid width
        grid_center = (xmlData.gridData.max + xmlData.gridData.min) * 0.5
        grid_width = xmlData.gridData.max - xmlData.gridData.min
        self.py_cell_count = np.ceil(grid_width / self.py_dx).astype(int)

        # recompute grid width, min and max
        grid_width = self.py_cell_count.astype(np.float32) * self.py_dx
        self.ti_grid_min = ti.Vector(grid_center - 0.5 * grid_width)
        self.ti_grid_max = ti.Vector(grid_center + 0.5 * grid_width)

        # allocating fields for grid mass and velocity (momentum)

        self.ti_grid_m = ti.field(float, self.py_cell_count)
        self.ti_grid_x = ti.Vector.field(3, float, self.py_cell_count)
        self.ti_grid_v = ti.Vector.field(3, float, self.py_cell_count)
        self.ti_grid_a = ti.Vector.field(3, float, self.py_cell_count)
        # for debug
        self.ti_grid_pos = ti.Vector.field(3, float, np.prod(self.py_cell_count))
        self.ti_grid_color = ti.field(ti.i32, np.prod(self.py_cell_count))

        #self.ti_particle_init_min = ti.Vector(xmlData.cuboidData.min)
        self.ti_particle_init_min = ti.Vector.field(3, float, 1)
        self.ti_particle_init_min.from_numpy(xmlData.cuboidData.min.astype(np.float32).reshape(1,3))
        self.py_particle_init_cell_samples_per_dim = xmlData.cuboidData.cell_samples_per_dim
        self.ti_particle_init_vel = ti.Vector(xmlData.cuboidData.vel)

        self.py_particle_hl = 0.5 * self.py_dx / xmlData.cuboidData.cell_samples_per_dim
        print('py_particle_hl: ', self.py_particle_hl)

        self.py_particle_volume = (self.py_dx / xmlData.cuboidData.cell_samples_per_dim)**3
        self.py_particle_mass = xmlData.cuboidData.density * self.py_particle_volume

        # initialize max number of particles
        cuboid_width = xmlData.cuboidData.max - xmlData.cuboidData.min
        self.ti_particle_ndcount = ti.field(int, 3)
        self.ti_particle_ndcount.from_numpy(np.ceil(cuboid_width * xmlData.cuboidData.cell_samples_per_dim / self.py_dx).astype(np.int32))
        # print('cuboid_width: ', cuboid_width)
        # print('xmlData.cuboidData.cell_samples_per_dim: ', xmlData.cuboidData.cell_samples_per_dim)
        # print('py_dx: ', self.py_dx)

        self.ti_particle_count = ti.field(int, ())
        self.ti_particle_count[None] = np.prod(self.ti_particle_ndcount.to_numpy())
        # print('ti_particle_count: ', self.ti_particle_count[None])
        self.ti_particle_is_inner_of_box = ti.field(int, self.ti_particle_count[None])
        self.ti_particle_x = ti.Vector.field(3, float, self.ti_particle_count[None])
        self.ti_particle_v = ti.Vector.field(3, float, self.ti_particle_count[None])
        self.ti_particle_be = ti.Matrix.field(3, 3, float, self.ti_particle_count[None])
        self.ti_particle_C = ti.Matrix.field(3, 3, float, self.ti_particle_count[None])
        # for debug
        # self.ti_particle_color_f = ti.field(float, self.ti_particle_count[None])
        # self.ti_particle_color = ti.field(float, self.ti_particle_count[None])

        self.changeHBParamKernel(xmlData.integratorData.herschel_bulkley_power, xmlData.integratorData.eta, xmlData.integratorData.yield_stress)

        _nd_count = np.ceil(cuboid_width * xmlData.cuboidData.cell_samples_per_dim / self.py_dx).astype(np.int32)
        print('changeSetUpData - cuboid_width: ', cuboid_width)
        print('changeSetUpData - cell_samples_per_dim: ', xmlData.cuboidData.cell_samples_per_dim)
        print('changeSetUpData - py_dx: ', self.py_dx)
        self.changeCuboid(xmlData.cuboidData.min[0], xmlData.cuboidData.min[1], xmlData.cuboidData.min[2], _nd_count[0].item(), _nd_count[1].item(), _nd_count[2].item())

        self.ti_num_boxes[None] = len(xmlData.staticBoxList)

        # staticBox_X = ti.field(float, ())
        # staticBox_X.from_numpy(width)
        self.changeSetUpDataKernel_Box0(xmlData.staticBoxList[0].min[0], xmlData.staticBoxList[0].min[1], xmlData.staticBoxList[0].min[2], xmlData.staticBoxList[0].max[0], xmlData.staticBoxList[0].max[1], xmlData.staticBoxList[0].max[2], int(xmlData.staticBoxList[0].isSticky))
        self.changeSetUpDataKernel_Box1(xmlData.staticBoxList[1].min[0], xmlData.staticBoxList[1].min[1], xmlData.staticBoxList[1].min[2], xmlData.staticBoxList[1].max[0], xmlData.staticBoxList[1].max[1], xmlData.staticBoxList[1].max[2], int(xmlData.staticBoxList[1].isSticky))
        self.changeSetUpDataKernel_Box2(xmlData.staticBoxList[2].min[0], xmlData.staticBoxList[2].min[1], xmlData.staticBoxList[2].min[2], xmlData.staticBoxList[2].max[0], xmlData.staticBoxList[2].max[1], xmlData.staticBoxList[2].max[2], int(xmlData.staticBoxList[2].isSticky))
        self.changeSetUpDataKernel_Box3(xmlData.staticBoxList[3].min[0], xmlData.staticBoxList[3].min[1], xmlData.staticBoxList[3].min[2], xmlData.staticBoxList[3].max[0], xmlData.staticBoxList[3].max[1], xmlData.staticBoxList[3].max[2], int(xmlData.staticBoxList[3].isSticky))

    @ti.kernel
    def changeSetUpDataKernel_Box0(self, box_0_min_x: ti.f32, box_0_min_y: ti.f32, box_0_min_z: ti.f32, box_0_max_x: ti.f32, box_0_max_y: ti.f32, box_0_max_z: ti.f32, isSticky: ti.i32):
        self.ti_static_box_min[0][0] = box_0_min_x
        self.ti_static_box_min[0][1] = box_0_min_y
        self.ti_static_box_min[0][2] = box_0_min_z
        self.ti_static_box_max[0][0] = box_0_max_x
        self.ti_static_box_max[0][1] = box_0_max_y
        self.ti_static_box_max[0][2] = box_0_max_z
        self.ti_static_box_type[0] = isSticky

    @ti.kernel
    def changeSetUpDataKernel_Box1(self, box_1_min_x: ti.f32, box_1_min_y: ti.f32, box_1_min_z: ti.f32, box_1_max_x: ti.f32, box_1_max_y: ti.f32, box_1_max_z: ti.f32, isSticky: ti.i32):
        self.ti_static_box_min[1][0] = box_1_min_x
        self.ti_static_box_min[1][1] = box_1_min_y
        self.ti_static_box_min[1][2] = box_1_min_z
        self.ti_static_box_max[1][0] = box_1_max_x
        self.ti_static_box_max[1][1] = box_1_max_y
        self.ti_static_box_max[1][2] = box_1_max_z
        self.ti_static_box_type[1] = isSticky

    @ti.kernel
    def changeSetUpDataKernel_Box2(self, box_2_min_x: ti.f32, box_2_min_y: ti.f32, box_2_min_z: ti.f32, box_2_max_x: ti.f32, box_2_max_y: ti.f32, box_2_max_z: ti.f32, isSticky: ti.i32):
        self.ti_static_box_min[2][0] = box_2_min_x
        self.ti_static_box_min[2][1] = box_2_min_y
        self.ti_static_box_min[2][2] = box_2_min_z
        self.ti_static_box_max[2][0] = box_2_max_x
        self.ti_static_box_max[2][1] = box_2_max_y
        self.ti_static_box_max[2][2] = box_2_max_z
        self.ti_static_box_type[2] = isSticky

    @ti.kernel
    def changeSetUpDataKernel_Box3(self, box_3_min_x: ti.f32, box_3_min_y: ti.f32, box_3_min_z: ti.f32, box_3_max_x: ti.f32, box_3_max_y: ti.f32, box_3_max_z: ti.f32, isSticky: ti.i32):
        self.ti_static_box_min[3][0] = box_3_min_x
        self.ti_static_box_min[3][1] = box_3_min_y
        self.ti_static_box_min[3][2] = box_3_min_z
        self.ti_static_box_max[3][0] = box_3_max_x
        self.ti_static_box_max[3][1] = box_3_max_y
        self.ti_static_box_max[3][2] = box_3_max_z
        self.ti_static_box_type[3] = isSticky

    @ti.kernel
    def changeCuboid(self, init_min_x: ti.f32, init_min_y: ti.f32, init_min_z: ti.f32, nd_count_x: ti.i32, nd_count_y: ti.i32, nd_count_z: ti.i32):
        self.ti_particle_init_min[0][0] = init_min_x
        self.ti_particle_init_min[0][1] = init_min_y
        self.ti_particle_init_min[0][2] = init_min_z

        self.ti_particle_ndcount[0] = nd_count_x
        self.ti_particle_ndcount[1] = nd_count_y
        self.ti_particle_ndcount[2] = nd_count_z
        self.ti_particle_count[None] = nd_count_x * nd_count_y * nd_count_z

        print('changeCuboid - ti_particle_count[None]: ', self.ti_particle_count[None])

    @ti.kernel
    def changeHBParamKernel(self, hb_n: ti.f32, hb_eta: ti.f32, hb_sigma_Y: ti.f32):
        self.ti_hb_n[None]      = hb_n
        self.ti_hb_eta[None]    = hb_eta
        self.ti_hb_sigmaY[None] = hb_sigma_Y

    def changeSetUpData(self, xmlData):
        self.changeHBParamKernel(xmlData.integratorData.herschel_bulkley_power, xmlData.integratorData.eta, xmlData.integratorData.yield_stress)

        cuboid_width = xmlData.cuboidData.max - xmlData.cuboidData.min
        _nd_count = np.ceil(cuboid_width * xmlData.cuboidData.cell_samples_per_dim / self.py_dx).astype(np.int32)
        print('changeSetUpData - cuboid_width: ', cuboid_width)
        print('changeSetUpData - cell_samples_per_dim: ', xmlData.cuboidData.cell_samples_per_dim)
        print('changeSetUpData - py_dx: ', self.py_dx)
        self.changeCuboid(xmlData.cuboidData.min[0], xmlData.cuboidData.min[1], xmlData.cuboidData.min[2], _nd_count[0].item(), _nd_count[1].item(), _nd_count[2].item())

        # self.ti_num_boxes[None] = len(xmlData.staticBoxList)
        
        # self.changeSetUpDataKernel_Box0(xmlData.staticBoxList[0].min[0], xmlData.staticBoxList[0].min[1], xmlData.staticBoxList[0].min[2], xmlData.staticBoxList[0].max[0], xmlData.staticBoxList[0].max[1], xmlData.staticBoxList[0].max[2], int(xmlData.staticBoxList[0].isSticky))
        # self.changeSetUpDataKernel_Box1(xmlData.staticBoxList[1].min[0], xmlData.staticBoxList[1].min[1], xmlData.staticBoxList[1].min[2], xmlData.staticBoxList[1].max[0], xmlData.staticBoxList[1].max[1], xmlData.staticBoxList[1].max[2], int(xmlData.staticBoxList[1].isSticky))
        self.changeSetUpDataKernel_Box2(xmlData.staticBoxList[2].min[0], xmlData.staticBoxList[2].min[1], xmlData.staticBoxList[2].min[2], xmlData.staticBoxList[2].max[0], xmlData.staticBoxList[2].max[1], xmlData.staticBoxList[2].max[2], int(xmlData.staticBoxList[2].isSticky))
        self.changeSetUpDataKernel_Box3(xmlData.staticBoxList[3].min[0], xmlData.staticBoxList[3].min[1], xmlData.staticBoxList[3].min[2], xmlData.staticBoxList[3].max[0], xmlData.staticBoxList[3].max[1], xmlData.staticBoxList[3].max[2], int(xmlData.staticBoxList[3].isSticky))

    @ti.kernel
    def initialize(self):
        self.ti_iteration[None] = 0
        # clear grid values
        for I in ti.grouped(self.ti_grid_m):
            self.ti_grid_m[I] = 0.0
            self.ti_grid_v[I] = ti.Vector.zero(float, 3)
            self.ti_grid_a[I] = ti.Vector.zero(float, 3)
            self.ti_grid_x[I] = self.ti_grid_min + I * self.py_dx

        # compute grid point locations (for debug)
        for i in range(self.py_cell_count[0]*self.py_cell_count[1]):
            gi = i % self.py_cell_count[0]
            gj = (i // self.py_cell_count[0]) % self.py_cell_count[1]
            gk = i // (self.py_cell_count[0] * self.py_cell_count[1])
            I = ti.Vector([gi, gj, gk])
            self.ti_grid_pos[i] = self.ti_grid_min + I.cast(float) * self.py_dx

        # initialize particles
        for i in range(self.ti_particle_count[None]):

            pi = i % self.ti_particle_ndcount[0]
            pj = (i // self.ti_particle_ndcount[0]) % self.ti_particle_ndcount[1]
            pk = i // (self.ti_particle_ndcount[0] * self.ti_particle_ndcount[1])
            # r = ti.Vector([ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5])
            r = ti.Vector([ 0.5, 0.5, 0.5])

            _I = ti.Vector([pi, pj, pk]).cast(float) + r
            self.ti_particle_x[i] = self.ti_particle_init_min[0] + (self.py_dx / self.py_particle_init_cell_samples_per_dim) * _I
            self.ti_particle_v[i] = self.ti_particle_init_vel
            self.ti_particle_be[i] = ti.Matrix.identity(float, 3)
            self.ti_particle_C[i] = ti.Matrix.zero(float,3, 3)
            self.ti_particle_is_inner_of_box[i] = 0


    # uGIMP basis functions
    @staticmethod
    @ti.func
    def linearIntegral(xp, hl, xi, w):
        diff = ti.abs(xp - xi)
        ret = 0.0
        if diff >= w + hl:
            ret = 0.0
        elif diff >= w - hl:
            ret = ((w + hl - diff) ** 2) / (2.0 * w)
        elif diff >= hl:
            ret = 2.0 * hl * (1.0 - diff / w)
        else:
            ret = 2.0 * hl - (hl * hl + diff * diff) / w
        return ret

    @staticmethod
    @ti.func
    def linearIntegralGrad(xp, hl, xi, w):
        diff = ti.abs(xp - xi)
        sgn = 1.0 if xp - xi >= 0.0 else -1.0
        ret = 0.0
        if diff >= w + hl:
            ret = 0.0
        elif diff >= w - hl:
            ret = -sgn * (w + hl - diff) / w
        elif diff >= hl:
            ret = -sgn * 2.0 * hl / w
        else:
            ret = 2.0 * (xi - xp) / w
        return ret

    @staticmethod
    @ti.func
    def uGIMPStencil():
        return ti.ndrange(3, 3, 3)

    @ti.func
    def uGIMPBase(self, particle_pos):
        return ((particle_pos - self.py_particle_hl - self.ti_grid_min) * self.py_invdx).cast(int)

    @ti.func
    def uGIMPWeightAndGrad(self, particle_pos, grid_pos):
        wx = self.linearIntegral(particle_pos[0], self.py_particle_hl, grid_pos[0], self.py_dx)
        wy = self.linearIntegral(particle_pos[1], self.py_particle_hl, grid_pos[1], self.py_dx)
        wz = self.linearIntegral(particle_pos[2], self.py_particle_hl, grid_pos[2], self.py_dx)
        weight = wx * wy * wz / self.py_particle_volume
        weight_grad = ti.Vector([wy * wz * self.linearIntegralGrad(particle_pos[0], self.py_particle_hl, grid_pos[0], self.py_dx), wx * wz * self.linearIntegralGrad(particle_pos[1], self.py_particle_hl, grid_pos[1], self.py_dx), wx * wy * self.linearIntegralGrad(particle_pos[2], self.py_particle_hl, grid_pos[2], self.py_dx)]) / self.py_particle_volume
        return weight, weight_grad

    @staticmethod
    @ti.func
    def bar_3d(A):
        return A / ti.pow(A.determinant(), 1.0/3.0)

    @staticmethod
    @ti.func
    def dev_3d(A):
        return A - (1.0/3.0) * A.trace() * ti.Matrix.identity(float, 3)

    @staticmethod
    @ti.func
    def hb_eval_3d(x, sigma_len_pre, mu_div_J, hb_sigma_y, hb_n, hb_eta, trace_be_bar, dt):
        return x - sigma_len_pre + ti.sqrt(2.0) * dt * mu_div_J * trace_be_bar * ti.pow( ( x / ti.sqrt(2.0) - hb_sigma_y ) / hb_eta, 1.0 / hb_n ) / 3.0

    @staticmethod
    @ti.func
    def hb_eval_deriv_3d(x, sigma_len_pre, mu_div_J, hb_sigma_y, hb_n, hb_eta, trace_be_bar, dt):
        return 1.0 + dt * mu_div_J * trace_be_bar * ti.pow( ( x / ti.sqrt(2.0) - hb_sigma_y ) / hb_eta, 1.0 / hb_n - 1.0 ) / (3.0 * hb_n * hb_eta)

    @ti.func
    def scalar_hb_solve_3d(self, sigma_len_pre, mu_div_J, hb_sigma_y, hb_n, hb_eta, trace_be_bar, dt):
        x = sigma_len_pre

        #while True:
        for i in range(14):
            fx = self.hb_eval_3d(x, sigma_len_pre, mu_div_J, hb_sigma_y, hb_n, hb_eta, trace_be_bar, dt)
            dfx = self.hb_eval_deriv_3d(x, sigma_len_pre, mu_div_J, hb_sigma_y, hb_n, hb_eta, trace_be_bar, dt)
            dx = - fx / dfx

            for j in range(20):
                x_new = x + dx
                if ( x_new / ti.sqrt(2.0) - hb_sigma_y ) >= 0:
                    x = x_new
                    break
                dx = dx / 2.0    

            if ti.abs(dx) < 1.0e-6:
                break

        return x

    @ti.func
    def isnan(self, x):
        return not (x < 0 or 0 < x or x == 0)

    @ti.kernel
    def step(self):
        self.ti_iteration[None] += 1

        # clear grid data
        for I in ti.grouped(self.ti_grid_m):
            self.ti_grid_m[I] = 0.0
            self.ti_grid_v[I] = ti.Vector.zero(float, 3)
            self.ti_grid_a[I] = ti.Vector.zero(float, 3)

        # particle status update and p2g
        for p in range(self.ti_particle_count[None]):
            base = self.uGIMPBase(self.ti_particle_x[p])
            stencil = self.uGIMPStencil()
            
            # compute particle stress
            J = ti.sqrt(self.ti_particle_be[p].determinant())
            be_bar = self.ti_particle_be[p] * pow(J, -2.0/3.0)
            dev_be_bar = be_bar - be_bar.trace() * ti.Matrix.identity(float, 3) / 3.0
            tau = self.py_kappa * 0.5 * (J+1.0) * (J-1.0) * ti.Matrix.identity(float, 3) + self.py_mu * dev_be_bar

            # p2g
            for i, j, k in ti.static(stencil):
                offset = ti.Vector([i, j, k])
                # grid point position
                gp = self.ti_grid_min + (base + offset).cast(float) * self.py_dx

                # compute weight and weight grad
                weight, weight_grad = self.uGIMPWeightAndGrad(self.ti_particle_x[p], gp)

                #internal force   
                f_internal = - self.py_particle_volume * tau @ weight_grad

                # accumulate grid velocity, acceleration and mass
                self.ti_grid_v[base + offset] += weight * self.py_particle_mass * ( self.ti_particle_v[p] + self.ti_particle_C[p] @ ( gp - self.ti_particle_x[p] ) )
                self.ti_grid_a[base + offset] += f_internal
                self.ti_grid_m[base + offset] += weight * self.py_particle_mass

        # grid update
        for I in ti.grouped(self.ti_grid_m):
            if self.ti_grid_m[I] > 0:
                old_momentum = self.ti_grid_v[I]
                new_momentum = old_momentum + self.py_dt * ( self.ti_grid_a[I] + self.ti_grid_m[I] * self.ti_g )

                # boundary conditions
                for s in range(self.ti_num_boxes[None]):
                    if self.ti_static_box_min[s][0] <= self.ti_grid_x[I][0] <= self.ti_static_box_max[s][0]:
                        if self.ti_static_box_min[s][1] <= self.ti_grid_x[I][1] <= self.ti_static_box_max[s][1]:
                            if self.ti_static_box_min[s][2] <= self.ti_grid_x[I][2] <= self.ti_static_box_max[s][2]:
                                new_momentum = ti.Vector.zero(float, 3)

                self.ti_grid_v[I] = new_momentum / self.ti_grid_m[I]
                self.ti_grid_a[I] = ( new_momentum - old_momentum ) / ( self.ti_grid_m[I] * self.py_dt )

        # g2p and update deformation status
        for p in range(self.ti_particle_count[None]):
            base = self.uGIMPBase(self.ti_particle_x[p])
            stencil = self.uGIMPStencil()

            v_pic = ti.Vector.zero(float, 3)
            grid_a = ti.Vector.zero(float, 3)
            vel_grad = ti.Matrix.zero(float, 3, 3)

            # compute velocity gradient and particle velocity
            for i, j, k in ti.static(stencil):
                offset = ti.Vector([i, j, k])
                # grid point position
                gp = self.ti_grid_min + (base + offset).cast(float) * self.py_dx

                # compute weight and weight grad
                weight, weight_grad = self.uGIMPWeightAndGrad(self.ti_particle_x[p], gp)

                vel_grad += self.ti_grid_v[base + offset].outer_product(weight_grad)
                v_pic += weight * self.ti_grid_v[base + offset]
                grid_a += weight * self.ti_grid_a[base + offset]

            self.ti_particle_v[p] = v_pic
            self.ti_particle_C[p] = vel_grad

            # elastic prediction
            f = ti.Matrix.identity(float, 3) + self.py_dt * vel_grad
            f_bar = self.bar_3d(f)
            be_bar = self.bar_3d(self.ti_particle_be[p])
            be_bar_pre = f_bar @ be_bar @ f_bar.transpose()

            be = f @ self.ti_particle_be[p] @ f.transpose()
            det_be = be.determinant()
            J = ti.sqrt(det_be)

            sigma_s_pre = self.py_mu * self.dev_3d(be_bar_pre) / J
            sigma_s_pre_len = sigma_s_pre.norm()

            scalar_sigma_pre = sigma_s_pre_len / ti.sqrt(2.0)

            # plastic correction
            if scalar_sigma_pre - self.ti_hb_sigmaY[None] > 0.0:
                sigma_s_pre_hat = sigma_s_pre / sigma_s_pre_len
                sigma_s_len = self.scalar_hb_solve_3d(sigma_s_pre_len, self.py_mu / J, self.ti_hb_sigmaY[None], self.ti_hb_n[None], self.ti_hb_eta[None], be_bar.trace(), self.py_dt)

                be_bar = (be_bar.trace() / 3.0) * ti.Matrix.identity(float, 3) + sigma_s_len * J * sigma_s_pre_hat / self.py_mu
                det_be_bar = be_bar.determinant()
                be = be_bar * ti.pow(det_be, 1.0/3.0) / ti.pow(det_be_bar, 1.0 / 3.0)


            self.ti_particle_be[p] = be
               
            # boundary conditions
            for s in range(self.ti_num_boxes[None]):
                if self.ti_static_box_min[s][0] <= self.ti_particle_x[p][0]<= self.ti_static_box_max[s][0]:
                    if self.ti_static_box_min[s][1] <= self.ti_particle_x[p][1] <= self.ti_static_box_max[s][1]:
                        if self.ti_static_box_min[s][2] <= self.ti_particle_x[p][2] <= self.ti_static_box_max[s][2]:
                            self.ti_particle_v[p] = ti.Vector.zero(float, 3)
                            self.ti_particle_C[p] = ti.Matrix.zero(float, 3, 3)
                            self.ti_particle_is_inner_of_box[p] = 1
                            break
                        else:
                            self.ti_particle_is_inner_of_box[p] = 0

            # advect
            self.ti_particle_x[p] += self.py_dt * self.ti_particle_v[p]

def T(a):
    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    c, s = np.cos(phi), np.sin(phi)
    C, S = np.cos(theta), np.sin(theta)
    x, z = x * c + z * s, z * c - x * s
    u, v = x, y * C + z * S
    return np.array([u, v]).swapaxes(0, 1) + 0.5

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



@ti.data_oriented
def MPM_simulation(n, eta, sigma_y, xmlData):

    xmlData.integratorData.herschel_bulkley_power = n
    xmlData.integratorData.eta = eta
    xmlData.integratorData.yield_stress = sigma_y

    x_diffs = []
    x_0frame = 0

    # output_dir = os.path.join(base_dir,f'setup_{height}_{width}')

    agtaichiMPM.changeSetUpData(xmlData)
    agtaichiMPM.initialize()


    agtaichiMPM.py_num_saved_frames = 0


    # os.makedirs(output_dir, exist_ok=True)
        
    print('*** Parameters ***')
    print('  herschel_bulkley_power: ' + str(agtaichiMPM.ti_hb_n[None]))
    print('  eta: ' + str(agtaichiMPM.ti_hb_eta[None]))
    print('  yield_stress: ' + str(agtaichiMPM.ti_hb_sigmaY[None]))
    # print('  setup width: ' + str(xmlData.cuboidData.max[0]))
    # print('  setup height: ' + str(xmlData.cuboidData.max[1]))

    while gui.running and not gui.get_event(gui.ESCAPE):
        for i in range(100):              
            agtaichiMPM.step()
            time = agtaichiMPM.ti_iteration[None] * agtaichiMPM.py_dt

            if time * agtaichiMPM.py_fps >= agtaichiMPM.py_num_saved_frames:

                particle_is_inner_of_box_id = np.where(agtaichiMPM.ti_particle_is_inner_of_box.to_numpy()[0:agtaichiMPM.ti_particle_count[None]].astype(np.int32) == 1)
                p_x = agtaichiMPM.ti_particle_x.to_numpy()[0:agtaichiMPM.ti_particle_count[None]].astype(np.float32)
                np.delete(p_x, particle_is_inner_of_box_id,axis=0)
                if agtaichiMPM.py_num_saved_frames == 0 :    
                    x_0frame = np.max(p_x[:, 0])
                    print('max x position: ', x_0frame)
                elif agtaichiMPM.py_num_saved_frames > 0:
                    x_diff = np.max(p_x[:, 0]) - x_0frame
                    x_diffs.append(x_diff)

                print("frame: ", agtaichiMPM.py_num_saved_frames)
                # fileOP.saveFile(agtaichiMPM, output_dir)
                agtaichiMPM.py_num_saved_frames += 1

                # memory_usage = process.memory_info().rss / 1024 ** 2
                # print(f"memory used: {memory_usage:.2f} MB")

                # pos = agtaichiMPM.ti_particle_x.to_numpy() / 20 + 0.3
                # gui.circles(T(pos), radius=2, color=0xFFFFFF)
                # gui.show()

        if agtaichiMPM.py_num_saved_frames > agtaichiMPM.py_max_frames:
            gc.collect()
            break 

    return np.array(x_diffs)



def run_optimization(n_iterations, xmlData):
    print("Running optimization...")

    optimizer = MultiTaskBayesianOptimizer(
        objective_fn=objective_wrapper,
        dimensions=3,
        bounds=BOUNDS,
        n_tasks=8,
        acquisition_type='EI',
        exploration_rate=0.1,
        random_state=42
    )

    start_time = time.time()
    history = optimizer.optimize(n_iter=n_iterations)
    optimization_time = time.time() - start_time

    print(f"Optimization completed in {optimization_time:.2f} seconds.")
    save_result(history)
    
    return history
    

def save_result(history, filename="optimization_results.csv"):

    df = pd.DataFrame({
        'n': history['X'][:, 0],
        'eta': history['X'][:, 1],
        'sigma_y': history['X'][:, 2]
    })
    
    
    for i in range(8):
        df[f'x_{i+1:02d}'] = history['Y'][:, i]
    
    
    best_values = np.max(history['best_values'], axis=0)
    for i in range(8):
        df[f'best_x_{i+1:02d}'] = best_values[i]
    
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mpm_bo_main.py <xml_config_file>")
        sys.exit(1)
    
    xmlData = xmlParser.MPMXMLData(sys.argv[1])
    gui = ti.GUI("MPM Simulation", res=(960, 540))
    fileOP = FileOperations()
    agtaichiMPM = AGTaichiMPM(xmlData)
    xmlData.show()

    width = 3.9
    height = 6.8
    
    run_optimization(n_iterations=10, xmlData=xmlData)
    
    gui.close()
    ti.reset()