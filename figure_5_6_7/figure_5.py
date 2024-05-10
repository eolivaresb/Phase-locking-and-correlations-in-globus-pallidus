###############################################################
################################################################
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
pi = np.pi
import pickle
################################################################
# Reverse the colormap to get the inverted version
cmap = plt.get_cmap('viridis')
inverted_cmap = cmap.reversed()
fz = 15
#####################################################
## aggregated data from save_aggregate_data.py
###############################################################
def load_data(name):
    with open('%s.pkl'%name, 'rb') as file:
        ldata = pickle.load(file)
    return [ldata['a1'], ldata['a2'], ldata['a3'], ldata['a4'], ldata['a5']]
###############################################################
################################################################
frequencies = np.arange(1, 101)
nfreq = len(frequencies)
###############################################################
[ph_hist, xgmean, CCamp, cc_amp, cc_ang] = load_data('data/data_exp')
nbins = 17
###############################################################

xph = np.arange(0.5/nbins, 1, 1./nbins)
plt.close('all')
fig = plt.figure(figsize = (10, 5))
gm = gridspec.GridSpec(180, 180, figure = fig)
ax22 = plt.subplot(gm[:, :87], projection ="3d")
ax12 = plt.subplot(gm[:, 93:], projection ="3d")

Xph, Y = np.zeros((nfreq, nbins)), np.ones((nfreq, nbins))
for k, f in enumerate(frequencies):
    Xph[k] = xph
    Y[k] *= f

fcut = 0
ph_cut = 0
clip = 6.9
ph_hist = np.array(ph_hist)
ph_hist[np.where(ph_hist> clip)] = clip

s1 = ax12.plot_surface(Xph[fcut:, ph_cut:], Y[fcut:, ph_cut:], np.array(ph_hist)[fcut:, ph_cut:], cmap=inverted_cmap, vmin = 0, vmax = 3.)
ax12.set_xlabel('CIF Phase', fontsize = fz)
ax12.set_ylabel('Stimulation frequency (Hz)', fontsize = fz)
ax12.set_zlabel('PDF', fontsize = fz)
ax12.set_zlim(0, 6.9)


# ax12.scatter(0.98, 30, 3., c='k', marker=r'$\ast$', s=17)
ax12.text(0.98, 26, 3., r'$\ast$', fontsize = fz)

s2 = ax22.plot_surface(np.array(xgmean)[fcut:], Y[fcut:] , np.array(CCamp)[fcut:], cmap=inverted_cmap, vmin = 0, vmax = .6, linewidth=0, antialiased=True)
ax22.set_xlabel('geometric mean\n (Spk/s)', fontsize = fz, labelpad = 10)
ax22.set_ylabel('Stimulation frequency (Hz)', fontsize = fz)
ax22.set_zlabel('CIF amplitude', fontsize = fz)
ax22.set_zlim(0, 1.09)
#######################

ax12.view_init(elev=35, azim= -10)  # Change these values as needed
ax22.view_init(elev=35, azim= -10)  # Change these values as needed

#######################
x1, x2, y1, fz = 0.01, 0.5, 0.95, 26
plt.figtext(x1, y1, 'A', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x2, y1, 'B', ha = 'left', va = 'center', fontsize = fz)
##################################################################

fig.subplots_adjust(left = 0.0, bottom = 0.06, right = 0.97, top = 1.00)
plt.savefig('./figure_5.png', dpi = 300)
