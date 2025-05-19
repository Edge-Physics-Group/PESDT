import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon
from los import LOS

class SynthDiag:

    def __init__(self, defs, diag, pulse=None, spec_line_dict=None, spec_line_dict_lytrap=None, use_AMJUEL = False):

        self.diag = diag
        self.pulse = pulse
        self.chords = []
        self.spec_line_dict = spec_line_dict
        self.spec_line_dict_lytrap = spec_line_dict_lytrap
        self.use_AMJUEL = use_AMJUEL
        
        self.get_spec_geom(defs)

    def get_spec_geom(self, defs):

        if self.diag in defs.diag_dict.keys():
            p2new = np.zeros((len(defs.diag_dict[self.diag]['p2']), 2))
            for i in range(len(defs.diag_dict[self.diag]['p1'])):
                r1 = defs.diag_dict[self.diag]['p1'][i, 0]
                z1 = defs.diag_dict[self.diag]['p1'][i, 1]
                r2 = defs.diag_dict[self.diag]['p2'][i, 0]
                z2 = defs.diag_dict[self.diag]['p2'][i, 1]
                w1 = defs.diag_dict[self.diag]['w'][i, 0]
                w2 = defs.diag_dict[self.diag]['w'][i, 1]
                if 'angle' in defs.diag_dict[self.diag]:
                    los_angle = defs.diag_dict[self.diag]['angle'][i]
                else:
                    los_angle = None
                theta = np.arctan2((r2 - r1), (z2 - z1))
                # elongate los to ensure defined LOS intersects machine wall (otherwise grid cells may be excluded)
                chord_L = np.sqrt((r1 - r2) ** 2 + (z1 - z2) ** 2)
                p2new[i, 0] = r2 + 1.0 * np.sin(theta)#*np.sign(theta)
                p2new[i, 1] = z2 + 1.0 * np.cos(theta)#*np.sign(theta)
                chord_L_elong = np.sqrt((r1 - p2new[i, 0]) ** 2 + (z1 - p2new[i, 1]) ** 2)
                w2_elong = w2 * chord_L_elong / chord_L
                self.chords.append(LOS(self.diag, los_poly=Polygon([(r1, z1),
                                                         (p2new[i, 0] - 0.5 * w2_elong * np.cos(theta),
                                                          p2new[i, 1] + 0.5 * w2_elong * np.sin(theta)),
                                                         (p2new[i, 0] + 0.5 * w2_elong * np.cos(theta),
                                                          p2new[i, 1] - 0.5 * w2_elong * np.sin(theta)),
                                                         (r1, z1)]),
                                       chord_num=defs.diag_dict[self.diag]['id'][i], p1=[r1, z1], w1=w1, p2orig=[r2, z2],
                                       p2=[p2new[i, 0], p2new[i, 1]], w2orig=w2, w2=w2_elong, l12=chord_L_elong, theta=theta,
                                       los_angle = los_angle, spec_line_dict=self.spec_line_dict,
                                       spec_line_dict_lytrap=self.spec_line_dict_lytrap, 
                                       use_AMJUEL = self.use_AMJUEL))

    def plot_LOS(self, ax, color='w', lw='2.0', Rrng=None):
        for chord in self.chords:
            if Rrng:
                if chord.v2unmod[0] >= Rrng[0] and chord.v2unmod[0] <= Rrng[1]:
                    los_patch = patches.Polygon(chord.los_poly.exterior.coords, closed=False, ec=color, lw=lw, fc='None', zorder=10)
                    ax.add_patch(los_patch)
            else:
                los_patch = patches.Polygon(chord.los_poly.exterior.coords, closed=False, ec=color, lw=lw, fc='None', zorder=10)
                ax.add_patch(los_patch)

    def plot_synth_spec_edge2d_data(self):
        fig, ax1 = plt.subplots(ncols=3, sharex=True, sharey=True)
        ax1[0].set_xlim(1.8, 4.0)
        ax1[0].set_ylim(-2.0, 2.0)
        fig.suptitle(self.diag)
        ax1[0].set_title(r'$\mathrm{T_{e}}$')
        ax1[1].set_title(r'$\mathrm{n_{e}}$')
        ax1[2].set_title(r'$\mathrm{n_{0}}$')

        recon_patches=[]
        recon_grid = []
        los_patches = []
        te=[]
        ne=[]
        n0=[]
        for chord in self.chords:
            los_patches.append(patches.Polygon(chord.los_poly.exterior.coords, closed=False, color='r'))
            for cell in chord.cells:
                recon_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True))
                recon_grid.append(patches.Polygon(cell.poly.exterior.coords, closed=False, color='k', alpha=1.0))
                te.append(cell.te)
                ne.append(cell.ne)
                n0.append(cell.n0)


        coll1 = PatchCollection(recon_patches)
        colors = plt.cm.jet(te/(np.max(te)/10.))
        coll1.set_color(colors)
        ax1[0].add_collection(coll1)

        coll2 = PatchCollection(recon_patches)
        colors = plt.cm.jet(ne/np.max(ne))
        coll2.set_color(colors)
        ax1[1].add_collection(coll2)

        coll3 = PatchCollection(recon_patches)
        colors = plt.cm.jet(n0/np.max(n0))
        coll3.set_color(colors)
        ax1[2].add_collection(coll3)

        split_grid = PatchCollection(recon_grid)
        split_grid.set_facecolor('None')
        split_grid.set_edgecolor('k')
        split_grid.set_linewidth(0.25)
        ax1[0].add_collection(split_grid)

        split_grid2 = PatchCollection(recon_grid)
        split_grid2.set_facecolor('None')
        split_grid2.set_edgecolor('k')
        split_grid2.set_linewidth(0.25)
        ax1[1].add_collection(split_grid2)

        split_grid3 = PatchCollection(recon_grid)
        split_grid3.set_facecolor('None')
        split_grid3.set_edgecolor('k')
        split_grid3.set_linewidth(0.25)
        ax1[2].add_collection(split_grid3)

        # los_coll = PatchCollection(los_patches)
        # los_coll.set_facecolor('None')
        # los_coll.set_edgecolor('r')
        # los_coll.set_linewidth(1.0)
        # ax1[0].add_collection(los_coll)
        #
        # los_coll1 = PatchCollection(los_patches)
        # los_coll1.set_facecolor('None')
        # los_coll1.set_edgecolor('r')
        # los_coll1.set_linewidth(1.0)
        # ax1[1].add_collection(los_coll1)
        #
        # los_coll2 = PatchCollection(los_patches)
        # los_coll2.set_facecolor('None')
        # los_coll2.set_edgecolor('r')
        # los_coll2.set_linewidth(1.0)
        # ax1[2].add_collection(los_coll2)

        # PLOT SEP INTERSECTIONS POINTS ON OUTER DIVERTOR LEG
        for chord in self.chords:
            if chord.shply_intersects_w_sep:
                ax1[0].plot(chord.shply_intersects_w_sep.coords.xy[0][0],chord.shply_intersects_w_sep.coords.xy[1][0], 'rx', ms=8, mew=3.0, zorder=10)

        if self.diag == 'KT3':
            # PLOT PLASMA PROPERTIES ALONG LOS
            fig2, ax2 = plt.subplots(nrows=4, sharex=True)
            ax2[0].set_xlim(0,6)
            for chord in self.chords:
                if chord.v2unmod[0] >=2.74 and chord.v2unmod[0]<=2.76:
                    col = np.random.rand(3,1)
                    ax2[0].plot(chord.los_1d['l'], chord.los_1d['te'], '-', color=col, lw=2.0)
                    ax2[1].plot(chord.los_1d['l'], chord.los_1d['ne'], '-', color=col, lw=2.0)
                    ax2[2].plot(chord.los_1d['l'], chord.los_1d['n0'], '-', color=col, lw=2.0)
                    for key in self.spec_line_dict['1']['1']:
                        ax2[3].plot(chord.los_1d['l'], np.asarray(chord.los_1d['H_emiss_per_vol'][key]['excit'])+np.asarray(chord.los_1d['H_emiss_per_vol'][key]['recom']), '-', color=col)

            ax2[0].set_ylabel(r'$\mathrm{T_{e}}$')
            ax2[1].set_ylabel(r'$\mathrm{n_{e}}$')
            ax2[2].set_ylabel(r'$\mathrm{n_{0}}$')
            ax2[3].set_ylabel(r'$\mathrm{ph\/s^{-1}\/m^{-2}\/sr^{-1}\/nm^{-1}}$')
            ax2[3].set_xlabel('distance along LOS (m)')

            # PLOT H INTEGRATED EMISSION (compare both methods for summing emission - expect identical results)
            fig3, ax3 = plt.subplots(nrows=1)
            # col_dict = {'6561.9':'r','4860.6':'m', '4339.9':'orange', '4101.2':'darkgreen', '3969.5':'b'}
            for key in self.spec_line_dict['1']['1']:
                emiss = []
                coord = []
                for chord in self.chords:
                    emiss.append(chord.los_int['H_emiss'][key]['excit'] + chord.los_int['H_emiss'][key]['recom'])
                    coord.append(chord.v2unmod[0])
                col = np.random.rand(3,1)
                ax3.semilogy(coord, emiss, '-', color=col, lw=3.0)
            ax3.set_xlabel('R tile 5 (m)')
            ax3.set_ylabel(r'$\mathrm{ph\/s^{-1}\/m^{-2}\/sr^{-1}\/nm^{-1}}$')
                # ax3.plot(chord.v2unmod[0], np.sum(chord.los_1d['H_emiss']['6561.9']['recom']), 'rx', markersize=10),