
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import patches
from math import atan

class MachineDefs:

    def __init__(self, name, wall_poly, pulse_ref=90531):
        self.name = name
        self.pulse_ref = pulse_ref
        self.__wall_poly = wall_poly
        self.__diag_dict = {}

    def set_diag_los(self, diag, los_geom):
        if diag not in self.__diag_dict.keys():
            self.__diag_dict[diag] = los_geom

    @property
    def regions(self):
        return self.__regions

    @property
    def diag_dict(self):
        return self.__diag_dict

    @property
    def wall_poly(self):
        return self.__wall_poly

def los_width_from_neigbh(los, los_neigh):
    los_m = (los[0,0]-los[1,0]) / (los[0,1]-los[1,1])
    los_neigh_m = (los_neigh[0,0]-los_neigh[1,0]) / (los_neigh[0,1]-los_neigh[1,1])

    los_mag = np.sqrt( (los[0,0]-los[1,0])**2 +  (los[0,1]-los[1,1])**2 )

    theta = np.arctan((los_m-los_neigh_m)/(1 + los_m*los_neigh_m))

    width = 2.* los_mag * np.sin(0.5*theta)

    return np.abs(width), theta / 2.0

def rotate_los(origin, p2, angle):
    p2_rot = p2 - origin
    # p2 = p2[:,None]
    rot_mat = np.array(([[np.cos(angle), -1.0*np.sin(angle)],[np.sin(angle), np.cos(angle)]]))

    p2_rot = rot_mat.dot(p2_rot)
    p2_rot+=origin

    return p2_rot

def poloidal_angle(p1, p2):

    # Convention: 0 deg on horizontal axis, rotation directed ccw

    opp = p2[1] - p1[1]
    adj = p2[0] - p1[0]
    try:
        theta = atan(opp/adj)*180./np.pi
    except ZeroDivisionError:
        if np.sign(opp)==1:
            pol_ang = 90.
        else:
            pol_ang = 270.
        return pol_ang

    # check quadrant and add 90/180/270 deg to theta
    if np.sign(opp)==-1 and np.sign(adj)==1:
        pol_ang = 360. - np.abs(theta)
    elif np.sign(opp)==-1 and np.sign(adj)==-1:
        pol_ang = 180. + np.abs(theta)
    elif np.sign(opp)==1 and np.sign(adj)==-1:
        pol_ang = 180. - np.abs(theta)
    else:
        pol_ang = 90.-theta

    return pol_ang

def get_JETdefs(plot_defs = False, pulse_ref = 90531):

    fwall = os.path.expanduser('~')+ '/PESDT/devices/JET/' + 'JET/wall.txt'
    wall_coords = np.genfromtxt(fwall, delimiter=' ')
    wall_poly = patches.Polygon(wall_coords, closed=False, ec='k', lw=2.0, fc='None', zorder=10)

    JET = MachineDefs('JET', wall_poly, pulse_ref = pulse_ref)

    # DIAGNOSTIC LOS DEFINITIONS: [r1, z1], [r2, z2], [w1, w2] for each LOS

    ###############
    # KT3A
    ###############
    # These defenitions don't match the ones in home/flush/surf/input/overlays_db.dat
    # --> rewrite KT3A, KT3B
    #origin = [3.28422, 3.56166]
    #width = 0.017226946
    #p2 = np.array(([2.55361, -1.59557],
    #               [2.57123, -1.60076],
    #               [2.58885, -1.60595],
    #               [2.60647, -1.61114],
    #               [2.62408, -1.61633],
    #               [2.64170, -1.62152],
    #               [2.65932, -1.62671],
    #               [2.67694, -1.63190],
    #               [2.69456, -1.63709],
    #               [2.71218, -1.64228],
    #               [2.72979, -1.64747],
    #               [2.74741, -1.65266],
    #               [2.76503, -1.65785],
    #               [2.78265, -1.66304],
    #               [2.80027, -1.66823],
    #               [2.81789, -1.67343],
    #               [2.83551, -1.67862],
    #               [2.85312, -1.68381],
    #               [2.87074, -1.68900],
    #               [2.88836, -1.69419],
    #               [2.90598, -1.69938],
    #               [2.92360, -1.70457]))
    #kt3a_los = np.zeros((len(p2), 3, 2))
    #for i in range(len(p2)):
    #    kt3a_los[i, 0] = origin
    #    kt3a_los[i, 1] = p2[i]
    #    kt3a_los[i, 2] = [0, width]
    #los_dict = {}
    #los_dict['p1'] = kt3a_los[:,0]
    #los_dict['p2'] = kt3a_los[:,1]
    #los_dict['w'] = kt3a_los[:,2]
    #los_dict['id'] = []
    #for id in range(len(kt3a_los)):
    #    los_dict['id'].append(str(id+1))
    #JET.set_diag_los('KT3', los_dict)
    ###############
    # KT3A
    ###############
    origin = [3.284220, 3.561660]
    width = 0.017226946 # Not recorded on overlays_db.dat ...
    p2 = np.array(([ 2.520510, -1.709300],
                   [2.544420, -1.673325],
                   [2.572691, -1.603562],
                   [2.590736, -1.606016],
                   [2.608511, -1.610387],
                   [2.626288, -1.614695],
                   [2.643699, -1.621916],
                   [2.661489, -1.626098],
                   [2.679163, -1.631209],
                   [2.696619, -1.638245],
                   [2.714313, -1.643232],
                   [2.732011, -1.648156],
                   [2.749715, -1.653018],
                   [2.767323, -1.658813],
                   [2.785036, -1.663549],
                   [2.802752, -1.668224],
                   [2.817116, -1.710689],
                   [2.835305, -1.711265],
                   [2.853547, -1.710780], 
                   [2.895780, -1.403168],
                   [2.914992, -1.374550],
                   [2.932983, -1.359822]))
    kt3a_los = np.zeros((len(p2), 3, 2))
    for i in range(len(p2)):
        kt3a_los[i, 0] = origin
        kt3a_los[i, 1] = p2[i]
        kt3a_los[i, 2] = [0, width]
    los_dict = {}
    los_dict['p1'] = kt3a_los[:,0]
    los_dict['p2'] = kt3a_los[:,1]
    los_dict['w'] = kt3a_los[:,2]
    los_dict['id'] = []
    for id in range(len(kt3a_los)):
        los_dict['id'].append(str(id+1))
    JET.set_diag_los('KT3A', los_dict)

    ###############
    # KT3B
    ###############
    origin = [3.284220, 3.561660]
    width = 0.017226946 # Not recorded on overlays_db.dat ...
    p2 = np.array(([ 2.533659, -1.687956],
                   [2.561793, -1.622243],
                   [2.582976, -1.603960],
                   [2.600988, -1.608390],
                   [2.619001, -1.612755],
                   [2.636645, -1.620032],
                   [2.654551, -1.625260],
                   [2.672582, -1.629431],
                   [2.690498, -1.634530],
                   [2.708201, -1.641553],
                   [2.726136, -1.646525],
                   [2.744075, -1.651432],
                   [2.761819, -1.658265],
                   [2.779868, -1.662049],
                   [2.797821, -1.666764],
                   [2.814704, -1.683367],
                   [2.830730, -1.710873],
                   [2.849136, -1.711421],
                   [2.891146, -1.412834], 
                   [2.911019, -1.379265],
                   [2.929409, -1.362573],
                   [2.947516, -1.347807]))
    kt3b_los = np.zeros((len(p2), 3, 2))
    for i in range(len(p2)):
        kt3b_los[i, 0] = origin
        kt3b_los[i, 1] = p2[i]
        kt3b_los[i, 2] = [0, width]
    los_dict = {}
    los_dict['p1'] = kt3b_los[:,0]
    los_dict['p2'] = kt3b_los[:,1]
    los_dict['w'] = kt3b_los[:,2]
    los_dict['id'] = []
    for id in range(len(kt3b_los)):
        los_dict['id'].append(str(id+1))
    JET.set_diag_los('KT3B', los_dict)

    ###############
    # KS3I
    ###############
    # KS3 inner divertor
    # R/z coords are from Surf, (/home/flush/surf/input/overlays_db.dat)
    origin = [3.33, 3.527]
    width = 0.033
    p2 = np.array(([2.29, -1.314],
                   [2.329, -1.334],
                   [2.367, -1.342],
                   [2.404, -1.389],
                   [2.414, -1.575],
                   [2.434, -1.713],
                   [2.481, -1.71],
                   [2.529, -1.695],
                   [2.585, -1.605],
                   [2.63, -1.617]))
    ks3i_los = np.zeros((len(p2), 3, 2))
    for i in range(len(p2)):
        ks3i_los[i, 0] = origin
        ks3i_los[i, 1] = p2[i]
        ks3i_los[i, 2] = [0, width]
    los_dict = {}
    los_dict['p1'] = ks3i_los[:,0]
    los_dict['p2'] = ks3i_los[:,1]
    los_dict['w'] = ks3i_los[:,2]
    los_dict['id'] = []
    for id in range(len(ks3i_los)):
        los_dict['id'].append(str(id+1))
    JET.set_diag_los('KS3I', los_dict)

    ###############
    # KS3O
    ###############
    # KS3 outer divertor
    # R/z coords are from Surf, (/home/flush/surf/input/overlays_db.dat)
    origin = [2.785, 3.527]
    width = 0.033
    p2 = np.array(([2.62, -1.614],
                   [2.665, -1.628],
                   [2.703, -1.641],
                   [2.744, -1.652],
                   [2.787, -1.665],
                   [2.831, -1.712],
                   [2.875, -1.716],
                   [2.911, -1.380],
                   [2.963, -1.336],
                   [3.020, -1.327]))
    ks3o_los = np.zeros((len(p2), 3,2))
    for i in range(len(p2)):
        ks3o_los[i, 0] = origin
        ks3o_los[i, 1] = p2[i]
        ks3o_los[i, 2] = [0, width]
    los_dict = {}
    los_dict['p1'] = ks3o_los[:,0]
    los_dict['p2'] = ks3o_los[:,1]
    los_dict['w'] = ks3o_los[:,2]
    los_dict['id'] = []
    for id in range(len(ks3o_los)):
        los_dict['id'].append(str(id+1))
    JET.set_diag_los('KS3O', los_dict)
         
    ###############
    # KS3H
    ###############
    origin = [6.130, 0.341]
    width = 0.095
    p2 = np.array(([1.768, 0.108],
                   [1.773, 0.196]))
    ks3h_los = np.zeros((len(p2), 3, 2))
    for i in range(len(p2)):
        ks3h_los[i, 0] = origin
        ks3h_los[i, 1] = p2[i]
        ks3h_los[i, 2] = [0, width]
    los_dict = {}
    los_dict['p1'] = ks3h_los[:,0]
    los_dict['p2'] = ks3h_los[:,1]
    los_dict['w'] = ks3h_los[:,2]
    los_dict['id'] = []
    for id in range(len(ks3h_los)):
        los_dict['id'].append(str(id+1))
    JET.set_diag_los('KS3H', los_dict)

    ###############
    # KS3J
    ###############
    origin = [6.130, 0.341]
    width = 0.095
    # p2 = np.array(([1.768, 0.108],
    #                [1.768, 0.110],
    #                [1.768, 0.112],
    #                [1.768, 0.196]))
    p2 = np.array(([1.768, 0.108],
                   [1.768, 0.196]))
    ks3j_los = np.zeros((len(p2), 3, 2))
    for i in range(len(p2)):
        ks3j_los[i, 0] = origin
        ks3j_los[i, 1] = p2[i]
        ks3j_los[i, 2] = [0, width]
    los_dict = {}
    los_dict['p1'] = ks3j_los[:,0]
    los_dict['p2'] = ks3j_los[:,1]
    los_dict['w'] = ks3j_los[:,2]
    los_dict['id'] = []
    for id in range(len(ks3j_los)):
        los_dict['id'].append(str(id+1))
    JET.set_diag_los('KS3J', los_dict)


    ###############
    # KS3V
    ###############
    origin = [3.115, 3.354]
    width = 0.095
    p2 = np.array(([3.251, -1.204],
                   [3.401, -1.204]))
    ks3v_los = np.zeros((len(p2), 3, 2))
    for i in range(len(p2)):
        ks3v_los[i, 0] = origin
        ks3v_los[i, 1] = p2[i]
        ks3v_los[i, 2] = [0, width]
    los_dict = {}
    los_dict['p1'] = ks3v_los[:,0]
    los_dict['p2'] = ks3v_los[:,1]
    los_dict['w'] = ks3v_los[:,2]
    los_dict['id'] = []
    for id in range(len(ks3v_los)):
        los_dict['id'].append(str(id+1))
    JET.set_diag_los('KS3V', los_dict)

    # ###############
    # # KS3I
    # ###############
    # origin = [3.115, 3.354]
    # width = 0.05
    # p2 = np.array(([3.101, -1.204],
    #                [3.151, -1.204],
    #                [3.201, -1.204],
    #                [3.251, -1.204],
    #                [3.301, -1.204],
    #                [3.351, -1.204],
    #                [3.401, -1.204]))
    # ks3i_los = np.zeros((len(p2), 3, 2))
    # for i in range(len(p2)):
    #     ks3i_los[i, 0] = origin
    #     ks3i_los[i, 1] = p2[i]
    #     ks3i_los[i, 2] = [0, width]
    # los_dict = {}
    # los_dict['p1'] = ks3i_los[:,0]
    # los_dict['p2'] = ks3i_los[:,1]
    # los_dict['w'] = ks3i_los[:,2]
    # los_dict['id'] = []
    # for id in range(len(ks3i_los)):
    #     los_dict['id'].append(str(id+1))
    # JET.set_diag_los('KS3I', los_dict)


    ###############
    # KT1J
    ###############
    # "Fake" KT1H diag
    origin = [6.238, -0.74]
    width = 0.3
    p2 = np.array(([1.94, 1.23],
                   [2.05, 1.54]))
    kt1j_los = np.zeros((len(p2), 3, 2))
    kt1j_los_angle = np.zeros((len(p2)))
    for i in range(len(p2)):
        kt1j_los[i, 0] = origin
        kt1j_los[i, 1] = p2[i]
        kt1j_los[i, 2] = [0, width]
        kt1j_los_angle[i] = 270. - np.arctan((-p2[i][0] + origin[0]) / ((-p2[i][1] + origin[1]))) * 180. / np.pi
    los_dict = {}
    los_dict['p1'] = kt1j_los[:,0]
    los_dict['p2'] = kt1j_los[:,1]
    los_dict['w'] = kt1j_los[:,2]
    los_dict['id'] = []
    for id in range(len(kt1j_los)):
        los_dict['id'].append(str(id+1))
    los_dict['angle'] = kt1j_los_angle
    JET.set_diag_los('KT1J', los_dict)


    ###############
    # KT1V
    ###############
    origin = [3.326, 3.807]
    p2 = np.array(([2.2787447, -1.31082201],
                   [2.30082607, -1.33440006],
                   [2.32782602, -1.33440006],
                   [2.35472608, -1.33440006],
                   [2.378052, -1.35385561],
                   [2.40063381, -1.37906277],
                   [2.41289091, -1.46512616],
                   [2.41917992, -1.59241283],
                   [2.42798877, -1.71315801],
                   [2.45742607, -1.70969999],
                   [2.48642612, -1.70969999],
                   [2.51542616, -1.70969999],
                   [2.55589104, -1.62998962],
                   [2.58804083, -1.60308468],
                   [2.61560178, -1.61040914],
                   [2.64301133, -1.6201551],
                   [2.67078233, -1.62739968],
                   [2.69836617, -1.63720798],
                   [2.72634029, -1.64494753],
                   [2.75428176, -1.65316343],
                   [2.78240013, -1.66152966],
                   [2.81055236, -1.66882849],
                   [2.83572602, -1.71150005],
                   [2.88960505, -1.41644132],
                   [2.92077351, -1.36994159],
                   [2.94975448, -1.34599328],
                   [2.97752619, -1.3348],
                   [3.00452614, -1.3348],
                   [3.02952623, -1.37039995],
                   [3.06067014, -1.29725635],
                   [3.08823347, -1.28003955]))
    kt1v_los = np.zeros((len(p2), 3, 2))
    kt1v_los_half_angle = np.zeros((len(p2)))
    kt1v_los_angle = np.zeros((len(p2)))
    # determine end point width using half angle between adjacent sight lines
    for i, los in enumerate(p2):
        if i > 0 and i < len(p2)-1:
            width, half_angle = los_width_from_neigbh(np.array((origin, p2[i])), np.array((origin, p2[i+1])))
            kt1v_los[i, 0] = origin
            kt1v_los[i, 1] = p2[i]
            kt1v_los[i, 2] = [0, width]
            kt1v_los_half_angle[i] = half_angle
        kt1v_los_angle[i] = 270.-np.arctan((-p2[i][0]+origin[0])/((-p2[i][1]+origin[1])))*180./np.pi
    # end points
    kt1v_los[0, 0] = origin
    kt1v_los[0, 1] = p2[0]
    kt1v_los[0, 2] = kt1v_los[1, 2]
    kt1v_los_half_angle[0] = kt1v_los_half_angle[1]
    kt1v_los[len(p2)-1, 0] = origin
    kt1v_los[len(p2)-1, 1] = p2[len(p2)-1]
    kt1v_los[len(p2)-1, 2] = kt1v_los[len(p2)-2, 2]
    kt1v_los_half_angle[len(p2)-1] = kt1v_los_half_angle[len(p2)-2]
    los_dict = {}
    los_dict['p1'] = kt1v_los[:,0]
    los_dict['p2'] = kt1v_los[:,1]
    los_dict['w'] = kt1v_los[:,2]
    los_dict['id'] = []
    for id in range(len(kt1v_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angle'] = kt1v_los_half_angle
    los_dict['angle'] = kt1v_los_angle
    JET.set_diag_los('KT1V', los_dict)

    ###############
    # KT1H
    ###############
    origin = [6.23800, -0.74000]
    p2 = np.array(([1.883, 0.838],
                   [1.888, 0.877],
                   [1.893, 0.915],
                   [1.896, 0.955],
                   [1.903, 0.993],
                   [1.909, 1.032],
                   [1.914, 1.072],
                   [1.921, 1.111],
                   [1.924, 1.151],
                   [1.934, 1.189],
                   [1.941, 1.228],
                   [1.926, 1.278],
                   [1.93, 1.321],
                   [1.971, 1.346],
                   [1.986, 1.384],
                   [2.001, 1.423],
                   [2.015, 1.461],
                   [2.03, 1.499],
                   [2.045, 1.538],
                   [2.059, 1.576],
                   [2.076, 1.613],
                   [2.094, 1.65],
                   [2.113, 1.686],
                   [2.131, 1.723],
                   [2.15, 1.76],
                   [2.169, 1.796],
                   [2.185, 1.835],
                   [2.236, 1.851],
                   [2.271, 1.876],
                   [2.309, 1.899],
                   [2.35, 1.92],
                   [2.394, 1.94],
                   [2.435, 1.954],
                   [2.476, 1.966],
                   [2.524, 1.974],
                   [2.57, 1.982],
                   [2.625, 1.983],
                   [2.682, 1.982],
                   [2.747, 1.973],
                   [2.826, 1.952]))
    kt1h_los = np.zeros((len(p2), 3, 2))
    kt1h_los_half_angle = np.zeros((len(p2)))
    kt1h_los_angle = np.zeros((len(p2)))
    # determine end point width using half angle between adjacent sight lines
    for i, los in enumerate(p2):
        if i > 0 and i < len(p2)-1:
            width, half_angle = los_width_from_neigbh(np.array((origin, p2[i])), np.array((origin, p2[i+1])))
            kt1h_los[i, 0] = origin
            kt1h_los[i, 1] = p2[i]
            kt1h_los[i, 2] = [0, width]
            kt1h_los_half_angle[i] = half_angle
        kt1h_los_angle[i] = 180-np.arctan(((p2[i][1]-origin[1]))/(-p2[i][0]+origin[0]))*180./np.pi
    # end points
    kt1h_los[0, 0] = origin
    kt1h_los[0, 1] = p2[0]
    kt1h_los[0, 2] = kt1h_los[1, 2]
    kt1h_los_half_angle[0] = kt1h_los_half_angle[1]
    kt1h_los[len(p2)-1, 0] = origin
    kt1h_los[len(p2)-1, 1] = p2[len(p2)-1]
    kt1h_los[len(p2)-1, 2] = kt1h_los[len(p2)-2, 2]
    kt1h_los_half_angle[len(p2)-1] = kt1h_los_half_angle[len(p2)-2]
    los_dict = {}
    los_dict['p1'] = kt1h_los[:,0]
    los_dict['p2'] = kt1h_los[:,1]
    los_dict['w'] = kt1h_los[:,2]
    los_dict['id'] = []
    for id in range(len(kt1h_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angle'] = kt1h_los_half_angle
    los_dict['angle'] = kt1h_los_angle
    JET.set_diag_los('KT1H', los_dict)

    ###############
    # KB5 (including pulse dependent configurations)
    ###############
    #Default vs. re-configured sight line config
    if JET.pulse_ref >=73758 and JET.pulse_ref <=82263:
        file = os.path.expanduser('~') + '/PESDT/JET/KB5_Bolometer_LOS_73758_82263.txt'
    else:
        file = os.path.expanduser('~') + '/PESDT/JET/KB5_Bolometer_LOS_default.txt'
    lines = np.genfromtxt(file, dtype=list, delimiter="\t", skip_header=3)

    # KB5
    kb5v_los = np.zeros((24, 3, 2))
    kb5v_los_half_angular_extent = np.zeros((24))
    kb5v_los_angle = np.zeros((24))
    kb5h_los = np.zeros((24, 3, 2))
    kb5h_los_half_angular_extent = np.zeros((24))
    kb5h_los_angle = np.zeros((24))
    # determine end point width using half angle between adjacent sight lines
    for i, los in enumerate(lines):
        if los[0] == b'KB5V':
            if int(los[1]) >= 1 and int(los[1]) <=24 and int(los[1]) != 8 \
                    and int(los[1]) != 16 and int(los[1]) != 24:
                origin = [float(lines[i][4]), float(lines[i][5])]
                p2 = [float(lines[i][6]), float(lines[i][7])]
                origin_neighb = [float(lines[i+1][4]), float(lines[i+1][5])]
                p2_neighb = [float(lines[i+1][6]), float(lines[i+1][7])]
            elif int(los[1]) == 8 or int(los[1]) == 16 or int(los[1]) == 24:
                origin = [float(lines[i][4]), float(lines[i][5])]
                p2 = [float(lines[i][6]), float(lines[i][7])]
                origin_neighb = [float(lines[i-1][4]), float(lines[i-1][5])]
                p2_neighb = [float(lines[i-1][6]), float(lines[i-1][7])]

            if int(los[1]) >= 1 and int(los[1]) <= 24:
                width, half_angle = los_width_from_neigbh(np.array((origin, p2)), np.array((origin_neighb, p2_neighb)))
                kb5v_los[i, 0] = origin
                kb5v_los[i, 1] = p2
                kb5v_los[i, 2] = [0, width]
                kb5v_los_half_angular_extent[i] = half_angle
                kb5v_los_angle[i] = los[8]
        if los[0] == b'KB5H':
            if int(los[1]) >= 1 and int(los[1]) <=24 and int(los[1]) != 8 and int(los[1]) != 24:
                origin = [float(lines[i][4]), float(lines[i][5])]
                p2 = [float(lines[i][6]), float(lines[i][7])]
                origin_neighb = [float(lines[i+1][4]), float(lines[i+1][5])]
                p2_neighb = [float(lines[i+1][6]), float(lines[i+1][7])]
            elif int(los[1]) == 8 or int(los[1]) == 24:
                origin = [float(lines[i][4]), float(lines[i][5])]
                p2 = [float(lines[i][6]), float(lines[i][7])]
                origin_neighb = [float(lines[i-1][4]), float(lines[i-1][5])]
                p2_neighb = [float(lines[i-1][6]), float(lines[i-1][7])]

            if int(los[1]) >= 1 and int(los[1]) <= 24:
                width, half_angle = los_width_from_neigbh(np.array((origin, p2)), np.array((origin_neighb, p2_neighb)))
                kb5h_los[i-32, 0] = origin
                kb5h_los[i-32, 1] = p2
                kb5h_los[i-32, 2] = [0, width]
                kb5h_los_half_angular_extent[i-32] = half_angle
                kb5h_los_angle[i-32] = los[8]

    los_dict = {}
    los_dict['p1'] = kb5v_los[:,0]
    los_dict['p2'] = kb5v_los[:,1]
    los_dict['w'] = kb5v_los[:,2]
    los_dict['id'] = []
    for id in range(len(kb5v_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angular_extent'] = kb5v_los_half_angular_extent
    los_dict['angle'] = kb5v_los_angle
    JET.set_diag_los('KB5V', los_dict)

    los_dict = {}
    los_dict['p1'] = kb5h_los[:,0]
    los_dict['p2'] = kb5h_los[:,1]
    los_dict['w'] = kb5h_los[:,2]
    los_dict['id'] = []
    for id in range(len(kb5h_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angle'] = kb5h_los_half_angular_extent
    los_dict['angle'] = kb5h_los_angle
    JET.set_diag_los('KB5H', los_dict)

    ###############
    # B3D4 (grouping of KB3303 and KB3304 channels)
    # Half angle for all chords: 5.85 deg
    ###############
    half_angle= 5.85 * np.pi / 180.

    file = os.path.expanduser('~') + '/PESDT/JET/B3D4_Bolometer_LOS.txt'
    lines = np.genfromtxt(file, dtype=list, delimiter="\t", skip_header=3)

    # B3D4
    B3D4_los = np.zeros((6, 3, 2))
    B3D4_los_half_angular_extent = np.zeros((6))
    B3D4_los_angle = np.zeros((6))
    B3D4_los_id = []
    # determine end point width using half angle between adjacent sight lines
    for i, los in enumerate(lines):
        origin = [float(los[2]), float(los[3])]
        p2 = [float(los[4]), float(los[5])]
        width = np.sqrt(np.abs(origin[0]-p2[0])**2+np.abs(origin[1]-p2[1])**2) * half_angle *np.pi/180.
        B3D4_los[i, 0] = origin
        B3D4_los[i, 1] = p2
        B3D4_los[i, 2] = [0, width]
        B3D4_los_half_angular_extent[i] = half_angle
        B3D4_los_angle[i] = los[6]
        B3D4_los_id.append(los[1].decode("utf-8"))

    los_dict = {}
    los_dict['p1'] = B3D4_los[:, 0]
    los_dict['p2'] = B3D4_los[:, 1]
    los_dict['w'] = B3D4_los[:, 2]
    los_dict['id'] = B3D4_los_id
    los_dict['half_angular_extent'] = B3D4_los_half_angular_extent
    los_dict['angle'] = B3D4_los_angle
    JET.set_diag_los('B3D4', los_dict)

    ###############
    # B3E4 (grouping of KB3301/302/307 channels)
    # Half angle for all chords: 5.85 deg
    ###############
    half_angle= 5.85 * np.pi / 180.

    file = os.path.expanduser('~') + '/PESDT/JET/B3E4_Bolometer_LOS.txt'
    lines = np.genfromtxt(file, dtype=list, delimiter="\t", skip_header=3)

    # B3D4
    B3E4_los = np.zeros((6, 3, 2))
    B3E4_los_half_angular_extent = np.zeros((6))
    B3E4_los_angle = np.zeros((6))
    B3E4_los_id = []
    # determine end point width using half angle between adjacent sight lines
    for i, los in enumerate(lines):
        origin = [float(los[2]), float(los[3])]
        p2 = [float(los[4]), float(los[5])]
        width = np.sqrt(np.abs(origin[0]-p2[0])**2+np.abs(origin[1]-p2[1])**2) * half_angle *np.pi/180.
        B3E4_los[i, 0] = origin
        B3E4_los[i, 1] = p2
        B3E4_los[i, 2] = [0, width]
        B3E4_los_half_angular_extent[i] = half_angle
        B3E4_los_angle[i] = los[6]
        B3E4_los_id.append(los[1].decode("utf-8"))

    los_dict = {}
    los_dict['p1'] = B3E4_los[:, 0]
    los_dict['p2'] = B3E4_los[:, 1]
    los_dict['w'] = B3E4_los[:, 2]
    los_dict['id'] = B3E4_los_id
    los_dict['half_angular_extent'] = B3E4_los_half_angular_extent
    los_dict['angle'] = B3E4_los_angle
    JET.set_diag_los('B3E4', los_dict)

    if plot_defs:
        plt.gca().add_patch(wall_poly)
        diag = 'B3E4'
        for i, los in enumerate(JET.diag_dict[diag]['id']):
            plt.plot([JET.diag_dict[diag]['p1'][i,0], JET.diag_dict[diag]['p2'][i, 0]],
                     [JET.diag_dict[diag]['p1'][i,1], JET.diag_dict[diag]['p2'][i, 1]],
                     '-k')
            p2_rot = rotate_los(JET.diag_dict[diag]['p1'][i],
                                JET.diag_dict[diag]['p2'][i], JET.diag_dict[diag]['half_angular_extent'][i])
            plt.plot([JET.diag_dict[diag]['p1'][i, 0], p2_rot[0]],
                    [JET.diag_dict[diag]['p1'][i, 1], p2_rot[1]], ':m')
            plt.text(1.7,1.7, diag, color='k')

        diag = 'B3D4'
        for i, los in enumerate(JET.diag_dict[diag]['id']):
            plt.plot([JET.diag_dict[diag]['p1'][i,0], JET.diag_dict[diag]['p2'][i, 0]],
                     [JET.diag_dict[diag]['p1'][i,1], JET.diag_dict[diag]['p2'][i, 1]],
                     '-k')
            p2_rot = rotate_los(JET.diag_dict[diag]['p1'][i],
                                JET.diag_dict[diag]['p2'][i], JET.diag_dict[diag]['half_angular_extent'][i])
            plt.plot([JET.diag_dict[diag]['p1'][i, 0], p2_rot[0]],
                    [JET.diag_dict[diag]['p1'][i, 1], p2_rot[1]], ':m')
            plt.text(1.7,1.7, diag, color='k')

        # for i, los in enumerate(JET.diag_dict['KT1V']['id']):
        #     plt.plot([JET.diag_dict['KT1V']['p1'][i,0], JET.diag_dict['KT1V']['p2'][i, 0]],
        #              [JET.diag_dict['KT1V']['p1'][i,1], JET.diag_dict['KT1V']['p2'][i, 1]],
        #              '-r')
        #     plt.text(1.7,1.7+0.2, 'KT1V', color='red')
        #     p2_rot = rotate_los(JET.diag_dict['KT1V']['p1'][i],
        #                         JET.diag_dict['KT1V']['p2'][i], kt1v_los_half_angle[i])
        #     plt.plot([JET.diag_dict['KT1V']['p1'][i,0], p2_rot[0]],
        #              [JET.diag_dict['KT1V']['p1'][i,1], p2_rot[1]], ':r')
        #     p2_rot2 = rotate_los(JET.diag_dict['KT1V']['p1'][i],
        #                          JET.diag_dict['KT1V']['p2'][i], -1.0*kt1v_los_half_angle[i])
        #     plt.plot([JET.diag_dict['KT1V']['p1'][i,0], p2_rot2[0]],
        #              [JET.diag_dict['KT1V']['p1'][i,1], p2_rot2[1]], ':r')

        # for i, los in enumerate(JET.diag_dict['KB5V']['id']):
        #     if i+1 >= 1 and i+1 <=24:
        #         plt.plot([JET.diag_dict['KB5V']['p1'][i, 0], JET.diag_dict['KB5V']['p2'][i, 0]],
        #                  [JET.diag_dict['KB5V']['p1'][i, 1], JET.diag_dict['KB5V']['p2'][i, 1]],
        #                  '-m')
        #         plt.text(1.7, 1.7 + 0.4, 'KB5V', color='m')
        #         p2_rot = rotate_los(JET.diag_dict['KB5V']['p1'][i],
        #                             JET.diag_dict['KB5V']['p2'][i], kb5v_los_half_angular_extent[i])
        #         #plt.plot([JET.diag_dict['KB5V']['p1'][i, 0], p2_rot[0]],
        #         #         [JET.diag_dict['KB5V']['p1'][i, 1], p2_rot[1]], ':m')
        #
        # for i, los in enumerate(JET.diag_dict['KB5H']['id']):
        #     if i+1 >= 1 and i+1 <=24:
        #         plt.plot([JET.diag_dict['KB5H']['p1'][i, 0], JET.diag_dict['KB5H']['p2'][i, 0]],
        #                  [JET.diag_dict['KB5H']['p1'][i, 1], JET.diag_dict['KB5H']['p2'][i, 1]],
        #                  '-g')
        #         plt.text(1.7, 1.7 + 0.6, 'KB5H', color='g')
        #         p2_rot = rotate_los(JET.diag_dict['KB5H']['p1'][i],
        #                             JET.diag_dict['KB5H']['p2'][i], kb5h_los_half_angular_extent[i])
        #         #plt.plot([JET.diag_dict['KB5H']['p1'][i, 0], p2_rot[0]],
        #         #         [JET.diag_dict['KB5H']['p1'][i, 1], p2_rot[1]], ':g')

        plt.axes().set_aspect('equal')
        plt.show()

    return JET


def get_DIIIDdefs(plot_defs=False):

    fwall = 'DIIID/d3d_efit_wall_174240.dat'
    wall_coords = np.genfromtxt(fwall, skip_header=3)
    wall_poly = patches.Polygon(wall_coords, closed=False, ec='k', lw=2.0, fc='None', zorder=10)


    DIIID = MachineDefs('DIIID', wall_poly)

    # DIAGNOSTIC LOS DEFINITIONS: [r1, z1], [r2, z2], [w1, w2] for each LOS

    ###############
    # bolo
    # bolo1: 1-15
    # bolo2: 16-24
    # bolo3: 25-35
    # bolo4: 36-48
    ###############
    bolo_ids = {
        'bolo1':np.arange(0,15),
        'bolo2':np.arange(15,24),
        'bolo3':np.arange(24,35),
        'bolo4':np.arange(35,48),
                }

    file = 'DIIID/bolo_geom_174240.dat'
    lines = np.genfromtxt(file, dtype=list, delimiter="\t", skip_header=3)

    bolo1_los = np.zeros((len(bolo_ids['bolo1']), 3, 2))
    bolo1_los_half_angular_extent = np.zeros((len(bolo_ids['bolo1'])))
    bolo1_los_angle = np.zeros((len(bolo_ids['bolo1'])))
    bolo2_los = np.zeros((len(bolo_ids['bolo2']), 3, 2))
    bolo2_los_half_angular_extent = np.zeros((len(bolo_ids['bolo2'])))
    bolo2_los_angle = np.zeros((len(bolo_ids['bolo2'])))
    bolo3_los = np.zeros((len(bolo_ids['bolo3']), 3, 2))
    bolo3_los_half_angular_extent = np.zeros((len(bolo_ids['bolo3'])))
    bolo3_los_angle = np.zeros((len(bolo_ids['bolo3'])))
    bolo4_los = np.zeros((len(bolo_ids['bolo4']), 3, 2))
    bolo4_los_half_angular_extent = np.zeros((len(bolo_ids['bolo4'])))
    bolo4_los_angle = np.zeros((len(bolo_ids['bolo4'])))

    # determine end point width using half angle between adjacent sight lines
    bolo_cams = ['bolo1', 'bolo2', 'bolo3', 'bolo4']
    for cam in bolo_cams:
        print(cam)
        for i, los in enumerate(bolo_ids[cam]):
            if los == bolo_ids[cam][-1]:
                neigh_index = los-1
            else:
                neigh_index = los+1
            origin = [float(lines[los][0]), float(lines[los][1])]
            p2 = [float(lines[los][2]), float(lines[los][3])]
            origin_neighb = [float(lines[neigh_index][0]), float(lines[neigh_index][1])]
            p2_neighb = [float(lines[neigh_index][2]), float(lines[neigh_index][3])]

            width, half_angle = los_width_from_neigbh(np.array((origin, p2)), np.array((origin_neighb, p2_neighb)))
            if cam == 'bolo1':
                bolo1_los[i, 0] = origin
                bolo1_los[i, 1] = p2
                bolo1_los[i, 2] = [0, width]
                bolo1_los_half_angular_extent[i] = half_angle
                bolo1_los_angle[i] = poloidal_angle(origin, p2)
            elif cam == 'bolo2':
                bolo2_los[i, 0] = origin
                bolo2_los[i, 1] = p2
                bolo2_los[i, 2] = [0, width]
                bolo2_los_half_angular_extent[i] = half_angle
                bolo2_los_angle[i] = poloidal_angle(origin, p2)
            elif cam == 'bolo3':
                bolo3_los[i, 0] = origin
                bolo3_los[i, 1] = p2
                bolo3_los[i, 2] = [0, width]
                bolo3_los_half_angular_extent[i] = half_angle
                bolo3_los_angle[i] = poloidal_angle(origin, p2)
            elif cam == 'bolo4':
                bolo4_los[i, 0] = origin
                bolo4_los[i, 1] = p2
                bolo4_los[i, 2] = [0, width]
                bolo4_los_half_angular_extent[i] = half_angle
                bolo4_los_angle[i] = poloidal_angle(origin, p2)

    los_dict = {}
    los_dict['p1'] = bolo1_los[:,0]
    los_dict['p2'] = bolo1_los[:,1]
    los_dict['w'] = bolo1_los[:,2]
    los_dict['id'] = []
    for id in range(len(bolo1_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angular_extent'] = bolo1_los_half_angular_extent
    los_dict['angle'] = bolo1_los_angle
    DIIID.set_diag_los('bolo1', los_dict)

    los_dict = {}
    los_dict['p1'] = bolo2_los[:,0]
    los_dict['p2'] = bolo2_los[:,1]
    los_dict['w'] = bolo2_los[:,2]
    los_dict['id'] = []
    for id in range(len(bolo2_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angular_extent'] = bolo2_los_half_angular_extent
    los_dict['angle'] = bolo2_los_angle
    DIIID.set_diag_los('bolo2', los_dict)

    los_dict = {}
    los_dict['p1'] = bolo3_los[:,0]
    los_dict['p2'] = bolo3_los[:,1]
    los_dict['w'] = bolo3_los[:,2]
    los_dict['id'] = []
    for id in range(len(bolo3_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angular_extent'] = bolo3_los_half_angular_extent
    los_dict['angle'] = bolo3_los_angle
    DIIID.set_diag_los('bolo3', los_dict)

    los_dict = {}
    los_dict['p1'] = bolo4_los[:,0]
    los_dict['p2'] = bolo4_los[:,1]
    los_dict['w'] = bolo4_los[:,2]
    los_dict['id'] = []
    for id in range(len(bolo4_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angular_extent'] = bolo4_los_half_angular_extent
    los_dict['angle'] = bolo4_los_angle
    DIIID.set_diag_los('bolo4', los_dict)

    ###############
    # bolo_hr (high resolution version)
    # bolo1: 1-15
    # bolo2: 16-24
    # bolo3: 25-35
    # bolo4: 36-48
    ###############
    bolo_hr_ids = {
        'bolo1_hr':np.arange(0,50),
        'bolo2_hr':np.arange(50,100),
        'bolo3_hr':np.arange(100,150),
        'bolo4_hr':np.arange(150,200),
                }

    file = 'DIIID/bolo_geom_174240_highres.dat'
    lines = np.genfromtxt(file, dtype=list, delimiter="\t", skip_header=3)

    bolo1_los = np.zeros((len(bolo_hr_ids['bolo1_hr']), 3, 2))
    bolo1_los_half_angular_extent = np.zeros((len(bolo_hr_ids['bolo1_hr'])))
    bolo1_los_angle = np.zeros((len(bolo_hr_ids['bolo1_hr'])))
    bolo2_los = np.zeros((len(bolo_hr_ids['bolo2_hr']), 3, 2))
    bolo2_los_half_angular_extent = np.zeros((len(bolo_hr_ids['bolo2_hr'])))
    bolo2_los_angle = np.zeros((len(bolo_hr_ids['bolo2_hr'])))
    bolo3_los = np.zeros((len(bolo_hr_ids['bolo3_hr']), 3, 2))
    bolo3_los_half_angular_extent = np.zeros((len(bolo_hr_ids['bolo3_hr'])))
    bolo3_los_angle = np.zeros((len(bolo_hr_ids['bolo3_hr'])))
    bolo4_los = np.zeros((len(bolo_hr_ids['bolo4_hr']), 3, 2))
    bolo4_los_half_angular_extent = np.zeros((len(bolo_hr_ids['bolo4_hr'])))
    bolo4_los_angle = np.zeros((len(bolo_hr_ids['bolo4_hr'])))

    # determine end point width using half angle between adjacent sight lines
    bolo_hr_cams = ['bolo1_hr', 'bolo2_hr', 'bolo3_hr', 'bolo4_hr']
    for cam in bolo_hr_cams:
        print(cam)
        for i, los in enumerate(bolo_hr_ids[cam]):
            if los == bolo_hr_ids[cam][-1]:
                neigh_index = los-1
            else:
                neigh_index = los+1
            origin = [float(lines[los][0]), float(lines[los][1])]
            p2 = [float(lines[los][2]), float(lines[los][3])]
            origin_neighb = [float(lines[neigh_index][0]), float(lines[neigh_index][1])]
            p2_neighb = [float(lines[neigh_index][2]), float(lines[neigh_index][3])]

            width, half_angle = los_width_from_neigbh(np.array((origin, p2)), np.array((origin_neighb, p2_neighb)))
            if cam == 'bolo1_hr':
                bolo1_los[i, 0] = origin
                bolo1_los[i, 1] = p2
                bolo1_los[i, 2] = [0, width]
                bolo1_los_half_angular_extent[i] = half_angle
                bolo1_los_angle[i] = poloidal_angle(origin, p2)
            elif cam == 'bolo2_hr':
                bolo2_los[i, 0] = origin
                bolo2_los[i, 1] = p2
                bolo2_los[i, 2] = [0, width]
                bolo2_los_half_angular_extent[i] = half_angle
                bolo2_los_angle[i] = poloidal_angle(origin, p2)
            elif cam == 'bolo3_hr':
                bolo3_los[i, 0] = origin
                bolo3_los[i, 1] = p2
                bolo3_los[i, 2] = [0, width]
                bolo3_los_half_angular_extent[i] = half_angle
                bolo3_los_angle[i] = poloidal_angle(origin, p2)
            elif cam == 'bolo4_hr':
                bolo4_los[i, 0] = origin
                bolo4_los[i, 1] = p2
                bolo4_los[i, 2] = [0, width]
                bolo4_los_half_angular_extent[i] = half_angle
                bolo4_los_angle[i] = poloidal_angle(origin, p2)

    los_dict = {}
    los_dict['p1'] = bolo1_los[:,0]
    los_dict['p2'] = bolo1_los[:,1]
    los_dict['w'] = bolo1_los[:,2]
    los_dict['id'] = []
    for id in range(len(bolo1_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angular_extent'] = bolo1_los_half_angular_extent
    los_dict['angle'] = bolo1_los_angle
    DIIID.set_diag_los('bolo1_hr', los_dict)

    los_dict = {}
    los_dict['p1'] = bolo2_los[:,0]
    los_dict['p2'] = bolo2_los[:,1]
    los_dict['w'] = bolo2_los[:,2]
    los_dict['id'] = []
    for id in range(len(bolo2_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angular_extent'] = bolo2_los_half_angular_extent
    los_dict['angle'] = bolo2_los_angle
    DIIID.set_diag_los('bolo2_hr', los_dict)

    los_dict = {}
    los_dict['p1'] = bolo3_los[:,0]
    los_dict['p2'] = bolo3_los[:,1]
    los_dict['w'] = bolo3_los[:,2]
    los_dict['id'] = []
    for id in range(len(bolo3_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angular_extent'] = bolo3_los_half_angular_extent
    los_dict['angle'] = bolo3_los_angle
    DIIID.set_diag_los('bolo3_hr', los_dict)

    los_dict = {}
    los_dict['p1'] = bolo4_los[:,0]
    los_dict['p2'] = bolo4_los[:,1]
    los_dict['w'] = bolo4_los[:,2]
    los_dict['id'] = []
    for id in range(len(bolo4_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angular_extent'] = bolo4_los_half_angular_extent
    los_dict['angle'] = bolo4_los_angle
    DIIID.set_diag_los('bolo4_hr', los_dict)

    ###############
    # filterscopes
    # fs1: 1-9
    # fs2: 10-18
    ###############
    fs_ids = {
        'fs1':np.arange(0,9),
        'fs2':np.arange(9,17)
    }

    file = 'DIIID/fs_geom_174240.dat'
    lines = np.genfromtxt(file, dtype=list, delimiter="\t", skip_header=3)

    fs1_los = np.zeros((len(fs_ids['fs1']), 3, 2))
    fs1_los_half_angular_extent = np.zeros((len(fs_ids['fs1'])))
    fs1_los_angle = np.zeros((len(fs_ids['fs1'])))

    fs2_los = np.zeros((len(fs_ids['fs2']), 3, 2))
    fs2_los_half_angular_extent = np.zeros((len(fs_ids['fs2'])))
    fs2_los_angle = np.zeros((len(fs_ids['fs2'])))

    # determine end point width using half angle between adjacent sight lines
    fs_cams = ['fs1', 'fs2']
    for cam in fs_cams:
        for i, los in enumerate(fs_ids[cam]):
            if los == fs_ids[cam][-1]:
                neigh_index = los-1
            else:
                neigh_index = los+1
            origin = [float(lines[los][0]), float(lines[los][1])]
            p2 = [float(lines[los][2]), float(lines[los][3])]
            origin_neighb = [float(lines[neigh_index][0]), float(lines[neigh_index][1])]
            p2_neighb = [float(lines[neigh_index][2]), float(lines[neigh_index][3])]

            width, half_angle = los_width_from_neigbh(np.array((origin, p2)), np.array((origin_neighb, p2_neighb)))

            if cam == 'fs1':
                fs1_los[i, 0] = origin
                fs1_los[i, 1] = p2
                fs1_los[i, 2] = [0, width]
                fs1_los_half_angular_extent[i] = half_angle
                fs1_los_angle[i] = poloidal_angle(origin, p2)
            elif cam == 'fs2':
                fs2_los[i, 0] = origin
                fs2_los[i, 1] = p2
                fs2_los[i, 2] = [0, width]
                fs2_los_half_angular_extent[i] = half_angle
                fs2_los_angle[i] = poloidal_angle(origin, p2)

    los_dict = {}
    los_dict['p1'] = fs1_los[:,0]
    los_dict['p2'] = fs1_los[:,1]
    los_dict['w'] = fs1_los[:,2]
    los_dict['id'] = []
    for id in range(len(fs1_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angular_extent'] = fs1_los_half_angular_extent
    los_dict['angle'] = fs1_los_angle
    DIIID.set_diag_los('fs1', los_dict)
    
    los_dict = {}
    los_dict['p1'] = fs2_los[:,0]
    los_dict['p2'] = fs2_los[:,1]
    los_dict['w'] = fs2_los[:,2]
    los_dict['id'] = []
    for id in range(len(fs2_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angular_extent'] = fs2_los_half_angular_extent
    los_dict['angle'] = fs2_los_angle
    DIIID.set_diag_los('fs2', los_dict)

    ###############
    # filterscopes, high resolution version
    # fs1: 1-20
    ###############
    fs_ids = {
        'fs1_hr': np.arange(0, 20),
    }

    file = 'DIIID/fs_geom_174240_highres.dat'
    lines = np.genfromtxt(file, dtype=list, delimiter="\t", skip_header=3)

    fs1_los = np.zeros((len(fs_ids['fs1_hr']), 3, 2))
    fs1_los_half_angular_extent = np.zeros((len(fs_ids['fs1_hr'])))
    fs1_los_angle = np.zeros((len(fs_ids['fs1_hr'])))

    # determine end point width using half angle between adjacent sight lines
    fs_hr_cams = ['fs1_hr']
    for cam in fs_hr_cams:
        for i, los in enumerate(fs_ids[cam]):
            if los == fs_ids[cam][-1]:
                neigh_index = los - 1
            else:
                neigh_index = los + 1
            origin = [float(lines[los][0]), float(lines[los][1])]
            p2 = [float(lines[los][2]), float(lines[los][3])]
            origin_neighb = [float(lines[neigh_index][0]), float(lines[neigh_index][1])]
            p2_neighb = [float(lines[neigh_index][2]), float(lines[neigh_index][3])]

            width, half_angle = los_width_from_neigbh(np.array((origin, p2)), np.array((origin_neighb, p2_neighb)))

            fs1_los[i, 0] = origin
            fs1_los[i, 1] = p2
            fs1_los[i, 2] = [0, width]
            fs1_los_half_angular_extent[i] = half_angle
            fs1_los_angle[i] = poloidal_angle(origin, p2)

    los_dict = {}
    los_dict['p1'] = fs1_los[:, 0]
    los_dict['p2'] = fs1_los[:, 1]
    los_dict['w'] = fs1_los[:, 2]
    los_dict['id'] = []
    for id in range(len(fs1_los)):
        los_dict['id'].append(str(id + 1))
    los_dict['half_angular_extent'] = fs1_los_half_angular_extent
    los_dict['angle'] = fs1_los_angle
    DIIID.set_diag_los('fs1_hr', los_dict)

    ###############
    # spectrometers
    # mds1: 1-8
    # mds2: 8-14
    ###############
    mds_ids = {
        'mds1': np.arange(0, 7),
        'mds2': np.arange(7, 14)
    }

    file = 'DIIID/mds_geom_174240.dat'
    lines = np.genfromtxt(file, dtype=list, delimiter="\t", skip_header=3)

    mds1_los = np.zeros((len(mds_ids['mds1']), 3, 2))
    mds1_los_half_angular_extent = np.zeros((len(mds_ids['mds1'])))
    mds1_los_angle = np.zeros((len(mds_ids['mds1'])))

    mds2_los = np.zeros((len(mds_ids['mds2']), 3, 2))
    mds2_los_half_angular_extent = np.zeros((len(mds_ids['mds2'])))
    mds2_los_angle = np.zeros((len(mds_ids['mds2'])))

    # determine end point width using half angle between adjacent sight lines
    mds_cams = ['mds1', 'mds2']
    for cam in mds_cams:
        for i, los in enumerate(mds_ids[cam]):
            if los == mds_ids[cam][-1]:
                neigh_index = los - 1
            else:
                neigh_index = los + 1
            origin = [float(lines[los][0]), float(lines[los][1])]
            p2 = [float(lines[los][2]), float(lines[los][3])]
            origin_neighb = [float(lines[neigh_index][0]), float(lines[neigh_index][1])]
            p2_neighb = [float(lines[neigh_index][2]), float(lines[neigh_index][3])]

            width, half_angle = los_width_from_neigbh(np.array((origin, p2)), np.array((origin_neighb, p2_neighb)))

            if cam == 'mds1':
                mds1_los[i, 0] = origin
                mds1_los[i, 1] = p2
                mds1_los[i, 2] = [0, width]
                mds1_los_half_angular_extent[i] = half_angle
                mds1_los_angle[i] = poloidal_angle(origin, p2)
            elif cam == 'mds2':
                mds2_los[i, 0] = origin
                mds2_los[i, 1] = p2
                mds2_los[i, 2] = [0, width]
                mds2_los_half_angular_extent[i] = half_angle
                mds2_los_angle[i] = poloidal_angle(origin, p2)

    los_dict = {}
    los_dict['p1'] = mds1_los[:, 0]
    los_dict['p2'] = mds1_los[:, 1]
    los_dict['w'] = mds1_los[:, 2]
    los_dict['id'] = []
    for id in range(len(mds1_los)):
        los_dict['id'].append(str(id + 1))
    los_dict['half_angular_extent'] = mds1_los_half_angular_extent
    los_dict['angle'] = mds1_los_angle
    DIIID.set_diag_los('mds1', los_dict)

    los_dict = {}
    los_dict['p1'] = mds2_los[:, 0]
    los_dict['p2'] = mds2_los[:, 1]
    los_dict['w'] = mds2_los[:, 2]
    los_dict['id'] = []
    for id in range(len(mds2_los)):
        los_dict['id'].append(str(id + 1))
    los_dict['half_angular_extent'] = mds2_los_half_angular_extent
    los_dict['angle'] = mds2_los_angle
    DIIID.set_diag_los('mds2', los_dict)

    ###############
    # spectrometer, high resolution version
    # mds1: 1-20
    ###############
    mds_ids = {
        'mds1_hr': np.arange(0, 20),
    }

    file = 'DIIID/mdslw_geom_174240_highres.dat'
    lines = np.genfromtxt(file, dtype=list, delimiter="\t", skip_header=3)

    mds1_los = np.zeros((len(mds_ids['mds1_hr']), 3, 2))
    mds1_los_half_angular_extent = np.zeros((len(mds_ids['mds1_hr'])))
    mds1_los_angle = np.zeros((len(mds_ids['mds1_hr'])))

    # determine end point width using half angle between adjacent sight lines
    mds_hr_cams = ['mds1_hr']
    for cam in mds_hr_cams:
        for i, los in enumerate(mds_ids[cam]):
            if los == mds_ids[cam][-1]:
                neigh_index = los - 1
            else:
                neigh_index = los + 1
            origin = [float(lines[los][0]), float(lines[los][1])]
            p2 = [float(lines[los][2]), float(lines[los][3])]
            origin_neighb = [float(lines[neigh_index][0]), float(lines[neigh_index][1])]
            p2_neighb = [float(lines[neigh_index][2]), float(lines[neigh_index][3])]

            width, half_angle = los_width_from_neigbh(np.array((origin, p2)), np.array((origin_neighb, p2_neighb)))

            mds1_los[i, 0] = origin
            mds1_los[i, 1] = p2
            mds1_los[i, 2] = [0, width]
            mds1_los_half_angular_extent[i] = half_angle
            mds1_los_angle[i] = poloidal_angle(origin, p2)

    los_dict = {}
    los_dict['p1'] = mds1_los[:, 0]
    los_dict['p2'] = mds1_los[:, 1]
    los_dict['w'] = mds1_los[:, 2]
    los_dict['id'] = []
    for id in range(len(mds1_los)):
        los_dict['id'].append(str(id + 1))
    los_dict['half_angular_extent'] = mds1_los_half_angular_extent
    los_dict['angle'] = mds1_los_angle
    DIIID.set_diag_los('mds1_hr', los_dict)


    ###############
    # divspred
    # divspred: 1-20
    ###############
    divspred_ids = {
        'divspred': np.arange(0, 20),
    }

    file = 'DIIID/divspred_geom_174240.dat'
    lines = np.genfromtxt(file, dtype=list, delimiter="\t", skip_header=4)

    divspred_los = np.zeros((len(divspred_ids['divspred']), 3, 2))
    divspred_los_half_angular_extent = np.zeros((len(divspred_ids['divspred'])))
    divspred_los_angle = np.zeros((len(divspred_ids['divspred'])))

    # determine end point width using half angle between adjacent sight lines
    divspred_cams = ['divspred']
    for cam in divspred_cams:
        for i, los in enumerate(divspred_ids[cam]):
            if los == divspred_ids[cam][-1]:
                neigh_index = los - 1
            else:
                neigh_index = los + 1
            origin = [float(lines[los][0]), float(lines[los][1])]
            p2 = [float(lines[los][2]), float(lines[los][3])]
            origin_neighb = [float(lines[neigh_index][0]), float(lines[neigh_index][1])]
            p2_neighb = [float(lines[neigh_index][2]), float(lines[neigh_index][3])]

            # This doesn't work for DIIID divspred since the origin moves along with R2, Z2
            width, half_angle = los_width_from_neigbh(np.array((origin, p2)), np.array((origin_neighb, p2_neighb)))

            divspred_los[i, 0] = origin
            divspred_los[i, 1] = p2
            divspred_los[i, 2] = [0, width]
            divspred_los_half_angular_extent[i] = half_angle
            divspred_los_angle[i] = poloidal_angle(origin, p2)

    los_dict = {}
    los_dict['p1'] = divspred_los[:, 0]
    los_dict['p2'] = divspred_los[:, 1]
    approx_width = 0.02 # TODO: get actual footprint
    los_dict['w'] = divspred_los[:, 2]+approx_width
    los_dict['id'] = []
    for id in range(len(divspred_los)):
        los_dict['id'].append(str(id + 1))
    # Need to manually add half angle because of special case where the origin is
    # shifted with the LOS R position, hence 0 angular extent
    divspred_los_half_angular_extent+=0.0037
    los_dict['half_angular_extent'] = divspred_los_half_angular_extent
    los_dict['angle'] = divspred_los_angle
    DIIID.set_diag_los('divspred', los_dict)


    if plot_defs:
        plt.gca().add_patch(wall_poly)
        diag = ['mds1_hr']#, 'bolo1', 'bolo2', 'bolo3', 'bolo4', 'fs1', 'fs2', 'mds1', 'mds2']
        # plt.gca().add_patch(wall_poly)
        colors = ['b', 'r', 'm', 'g','k', 'orange', 'brown', 'pink']
        for icam, cam in enumerate(diag):
            for i, los in enumerate(DIIID.diag_dict[cam]['id']):
                plt.plot([DIIID.diag_dict[cam]['p1'][i, 0], DIIID.diag_dict[cam]['p2'][i, 0]],
                         [DIIID.diag_dict[cam]['p1'][i, 1], DIIID.diag_dict[cam]['p2'][i, 1]],
                         '-', c=colors[icam])
                plt.text(0.7, -0.9 + 0.1* icam, cam, color=colors[icam])
                p2_rot = rotate_los(DIIID.diag_dict[cam]['p1'][i],
                                    DIIID.diag_dict[cam]['p2'][i], DIIID.diag_dict[cam]['half_angular_extent'][i])
                plt.plot([DIIID.diag_dict[cam]['p1'][i, 0], p2_rot[0]],
                        [DIIID.diag_dict[cam]['p1'][i, 1], p2_rot[1]], ':', c=colors[icam])

        plt.axes().set_aspect('equal')
        plt.show()

        # plot against poloidal angle
        for icam, cam in enumerate(diag):
            print(cam)
            for i, los in enumerate(DIIID.diag_dict[cam]['id']):
                print(los, DIIID.diag_dict[cam]['angle'][i])
                plt.plot(int(los), DIIID.diag_dict[cam]['angle'][i], 'o', c=colors[icam])
                plt.text(0.7, 0.7 + 0.4, cam, color=colors[icam])

        plt.show()
    return DIIID

if __name__=='__main__':

    print('Contains some useful geometry and instrument definitions for supported experimental devices.')
