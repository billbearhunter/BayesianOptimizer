import numpy as np
import xml.etree.ElementTree as ET


class MPMXMLStaticBoxData:
    def __init__(self, dim):
        self.min = np.zeros(dim)
        self.max = np.zeros(dim)
        self.isSticky = False

    def setFromXMLNode(self, node):
        self.min = np.array([float(e) for e in node.attrib['min'].split(' ')])
        self.max = np.array([float(e) for e in node.attrib['max'].split(' ')])
        if node.attrib['boundary_behavior'] == 'sticking':
            self.isSticky = True
        else:
            self.isSticky = False

    def show(self):
        print('*** Static box ***')
        print('  min: ' + str(self.min))
        print('  max: ' + str(self.max))
        print('  isSticky: ' + str(self.isSticky))


class MPMXMLparticleSkinnerData:
    def __init__(self):
        self.path = ""
        self.grid_space = 0.0
        self.file_type = ""

    def setFromXMLNode(self, node):
        self.path = str(node.attrib['path'])
        self.grid_space = float(node.attrib['grid_space'])
        self.file_type = str(node.attrib['file_type'])


    def show(self):
        print('*** Particle Skinner Arguments ***')
        # print('  path      : ' + str(self.path))
        print('  grid_space: ' + str(self.grid_space))
        print('  file_type : ' + str(self.file_type))


class MPMXMLGLRenderData:

    class MPMXMLCameraData:
        def __init__(self):
            self.eyepos = np.zeros(3)
            self.quat = np.zeros(4)
            self.window_size = np.zeros(2)
            self.fov = 0.0

        def setFromXMLNode(self, node):
            self.eyepos = np.array([float(e) for e in node.attrib['eyepos'].split(' ')])
            self.quat = np.array([float(e) for e in node.attrib['quat'].split(' ')])
            self.window_size = np.array([float(e) for e in node.attrib['window_size'].split(' ')])
            self.fov = float(node.attrib['fov'])

        # def show(self):
        #     print('  camera :')
        #     print('    eyepos      : ' + str(self.eyepos))
        #     print('    quat: ' + str(self.quat))
        #     print('    window_size : ' + str(self.window_size))
        #     print('    fov : ' + str(self.fov))

    def __init__(self):
        self.path = ""
        self.cameraData = self.MPMXMLCameraData()

    def setFromXMLNode(self, node):
        self.path = str(node.attrib['path'])
        self.cameraData.setFromXMLNode(node.find('camera'))

    # def show(self):
    #     print('*** GLRender Arguments ***')
    #     print('  path      : ' + str(self.path))
    #     self.cameraData.show()

class MPMXMLIntegratorData:
    def __init__(self):
        self.dt = 0.0
        self.bulk_modulus = 1.0 
        self.shear_modulus = 1.0 
        self.flip_pic_alpha = 0.95

        self.herschel_bulkley_power = 0.3
        self.eta = 0.0001 
        self.yield_stress = 0.0 


        self.fps = 50
        self.max_frames = 4


    def setFromXMLNode(self, node):
        # self.herschel_bulkley_power = float(node.attrib['herschel_bulkley_power'])
        # self.eta = float(node.attrib['eta'])
        # self.yield_stress = float(node.attrib['yield_stress'])

        self.dt = float(node.attrib['dt'])
        self.bulk_modulus = float(node.attrib['bulk_modulus'])
        self.shear_modulus = float(node.attrib['shear_modulus'])
        self.flip_pic_alpha = float(node.attrib['flip_pic_alpha'])

        self.fps = int(node.attrib['fps'])
        self.max_frames = int(node.attrib['max_frames'])

    def show(self):
        print('*** Integrator ***')
        print('  dt: ' + str(self.dt))
        print('  bulk_modulus: ' + str(self.bulk_modulus))
        print('  shear_modulus: ' + str(self.shear_modulus))
        print('  flip_pic_alpha: ' + str(self.flip_pic_alpha))

        print('  fps: ' + str(self.fps))
        print('  max_frames: ' + str(self.max_frames))

        print('  herschel_bulkley_power: ' + str(self.herschel_bulkley_power))
        print('  eta: ' + str(self.eta))
        print('  yield_stress: ' + str(self.yield_stress))

class MPMXMLGridData:
    def __init__(self, dim):
        self.min = np.zeros(dim)
        self.max = np.zeros(dim)
        self.cell_width = 1.0

    def setFromXMLNode(self, node):
        self.min = np.array([float(e) for e in node.attrib['min'].split(' ')])
        self.max = np.array([float(e) for e in node.attrib['max'].split(' ')])
        self.cell_width = float(node.attrib['cell_width'])

    def show(self):
        print('*** Grid ***')
        print('  min: ' + str(self.min))
        print('  max: ' + str(self.max))
        print('  cell_width: ' + str(self.cell_width))

class MPMXMLCuboidData:
    def __init__(self, dim):
        self.min = np.zeros(dim)
        self.max = np.zeros(dim)
        self.density = 1.0
        self.cell_samples_per_dim = 2
        self.vel = np.zeros(dim)

    def setFromXMLNode(self, node):
        self.min = np.array([float(e) for e in node.attrib['min'].split(' ')])
        self.max = np.array([float(e) for e in node.attrib['max'].split(' ')])
        self.density = float(node.attrib['density'])
        self.cell_samples_per_dim = int(node.attrib['cell_samples_per_dim'])
        self.vel = np.array([float(e) for e in node.attrib['vel'].split(' ')])

    def show(self):
        print('*** Cuboid ***')
        print('  min: ' + str(self.min))
        print('  max: ' + str(self.max))
        print('  density: ' + str(self.density))
        print('  cell_samples_per_dim: ' + str(self.cell_samples_per_dim))
        print('  vel: ' + str(self.vel))

class MPMXMLStaticPlaneData:
    def __init__(self, dim):
        self.x = np.zeros(dim)
        self.n = np.zeros(dim)
        self.isSticky = False

    def setFromXMLNode(self, node):
        self.x = np.array([float(e) for e in node.attrib['x'].split(' ')])
        self.n = np.array([float(e) for e in node.attrib['n'].split(' ')])
        if node.attrib['boundary_behavior'] == 'sticking':
            self.isSticky = True
        else:
            self.isSticky = False

    def show(self):
        print('*** Static plane ***')
        print('  x: ' + str(self.x))
        print('  n: ' + str(self.n))
        print('  isSticky: ' + str(self.isSticky))

class MPMXMLNearEarthGravityData:
    def __init__(self, dim):
        self.g = np.zeros(dim)

    def setFromXMLNode(self, node):
        self.g = np.array([float(e) for e in node.attrib['f'].split(' ')])

    def show(self):
        print('*** Near earth gravity ***')
        print('  g: ' + str(self.g))

class MPMXMLData:
    def __init__(self, file_name):
        print('[AGTaichiMPM3D] Parsing xml file: ' + str(file_name))
        tree = ET.parse(file_name)
        root = tree.getroot()
        if root.tag != 'AGTaichiMPM3D':
            print('[AGTaichiMPM3D] Could not find root note AGTaichiMPM3D. Exiting...')
            exit(-1)

        self.staticBoxList = []
        for static_box in root.findall('static_box'):
            staticBoxData = MPMXMLStaticBoxData(3)
            staticBoxData.setFromXMLNode(static_box)
            self.staticBoxList.append(staticBoxData)
            

        ps = root.find('particle_skinner')
        self.particleSkinnerData = MPMXMLparticleSkinnerData()
        self.particleSkinnerData.setFromXMLNode(ps)

        GLRender = root.find('GLRender')
        self.GLRenderData = MPMXMLGLRenderData()
        self.GLRenderData.setFromXMLNode(GLRender)
        
        integrator = root.find('integrator')
        self.integratorData = MPMXMLIntegratorData()
        self.integratorData.setFromXMLNode(integrator)

        grid = root.find('grid')
        self.gridData = MPMXMLGridData(3)
        self.gridData.setFromXMLNode(grid)

        cuboid = root.find('cuboid')
        self.cuboidData = MPMXMLCuboidData(3)
        self.cuboidData.setFromXMLNode(cuboid)

        nearEarthGravity = root.find('near_earth_gravity')
        self.nearEarthGravityData = MPMXMLNearEarthGravityData(3)
        self.nearEarthGravityData.setFromXMLNode(nearEarthGravity)


    def show(self):
        print('[AGTaichiMPM3D] XML Data:')
        self.particleSkinnerData.show()
        # self.GLRenderData.show()
        self.integratorData.show()
        self.cuboidData.show()
        for sb in self.staticBoxList:
            sb.show()

