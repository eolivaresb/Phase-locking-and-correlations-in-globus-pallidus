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
def save_data(name, ph_hist, xgmean, CC_ampl, cc_amp, cc_ang):
    data_to_save = {'a1': ph_hist, 'a2': xgmean, 'a3': CC_ampl, 'a4': cc_amp, 'a5': cc_ang}
    with open('%s.pkl'%name, 'wb') as file:
        pickle.dump(data_to_save, file)
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
###############################################################
[ph_hist, xgmean, CCamp, cc_amp, cc_ang] = load_data('data/data_exp')
[ph_hist_control, xgmean_control, CC_ampl_control, cc_amp_control, cc_ang_control] = load_data('data/data_control')
###############################################################
nbins = 17
ph_hist_control = np.array(ph_hist_control)
###############################################################
xph = np.arange(0.5/nbins, 1, 1./nbins)
plt.close('all')
fig = plt.figure(figsize = (10, 7.5))
gm = gridspec.GridSpec(140, 180, figure = fig)

ax21 = plt.subplot(gm[:90, :87], projection ="3d")
ax11 = plt.subplot(gm[:90, 93:], projection ="3d")

ax23 = plt.subplot(gm[108:, 15:85])
ax13 = plt.subplot(gm[108:, 110:])
###############################################################
###############################################################
Xph, Y = np.zeros((nfreq, nbins)), np.ones((nfreq, nbins))
for k, f in enumerate(frequencies):
    Xph[k] = xph
    Y[k] *= f

fcut = 0
ph_cut = 0
clip = 5.4
ph_hist_control = np.array(ph_hist_control)
ph_hist_control[np.where(ph_hist_control> clip)] = clip
###############################################################
ax11.plot_surface(Xph[fcut:, ph_cut:], Y[fcut:, ph_cut:], ph_hist_control[fcut:, ph_cut:], cmap=inverted_cmap, vmin = 0, vmax = 3.)
for ax in [ax11]:
    ax.set_xlabel('CIF Phase', fontsize = fz)
    ax.set_ylabel('Stimulation frequency (Hz)', fontsize = fz)
    ax.set_zlabel('PDF', fontsize = fz)
    ax.set_zlim(0, clip)

ax21.plot_surface(np.array(xgmean_control)[fcut:], Y[fcut:] , np.array(CC_ampl_control)[fcut:], cmap=inverted_cmap, vmin = 0, vmax = .6, linewidth=0, antialiased=True)
for ax in [ax21]:
    ax.set_xlabel('geometric mean\n (Spk/s)', fontsize = fz)
    ax.set_ylabel('Stimulation frequency (Hz)', fontsize = fz)
    ax.set_zlabel('CIF amplitude', fontsize = fz)
    ax.set_zlim(0, 1.09)
#######################
for ax in [ax11, ax21]:
    ax.view_init(elev=35, azim= -10)  # Change these values as needed
###############################################################
def plot_m(ax, color, data, label):
    mean, std = data
    ax.plot(frequencies, mean, lw = 2, color = color, label = label)
    ax.fill_between(frequencies, mean-std, mean+std, lw = 0, color = color, alpha = 0.13)

plot_m(ax13, 'r', cc_ang,  'Data')
plot_m(ax13, 'b', cc_ang_control,  'Uncoupled GP')

plot_m(ax23, 'r', cc_amp,  'Data')
plot_m(ax23, 'b', cc_amp_control,  'Uncoupled GP')

for ax in [ax13, ax23]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(9.5, 100.5)
    ax.set_xlabel('Stimulus frequency (Hz)', fontsize = fz)

ax13.set_ylabel('CIF phase', fontsize = fz)
ax23.set_ylabel('CIF amplitude', fontsize = fz)

ax13.legend(bbox_to_anchor=(.9, 1.2), frameon = False, fontsize = 13, ncol = 2)
###############################################################
###############################################################
x1, x2, y1, y2, y3, fz = 0.01, 0.515, 0.95, 0.31, 0.21, 26
plt.figtext(x1, y1, 'A', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x2, y1, 'B', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x1, y2, 'C', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x2, y2, 'D', ha = 'left', va = 'center', fontsize = fz)
##################################################################

fig.subplots_adjust(left = 0.0, bottom = 0.07, right = 0.97, top = 1.00)
plt.savefig('./figure_6.png', dpi = 300)
###############################################################
