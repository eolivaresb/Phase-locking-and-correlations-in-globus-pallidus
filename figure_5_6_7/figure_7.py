################################################################
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import pickle
pi = np.pi
################################################################
# Reverse the colormap to get the inverted version
cmap = plt.get_cmap('viridis')
inverted_cmap = cmap.reversed()
fz = 15
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
#### Load aggregated data
###############################################################
[ph_hist_sworld, xgmean_sworld, CC_ampl_sworld, cc_amp_sworld, cc_ang_sworld] = load_data('data/data_sworld')
[ph_hist_control, xgmean_control, CC_ampl_control, cc_amp_control, cc_ang_control] = load_data('data/data_barrage')
###############################################################
nbins = 17
ph_hist_sworld = np.array(ph_hist_sworld)
ph_hist_control = np.array(ph_hist_control)
xph = np.arange(0.5/nbins, 1, 1./nbins)
###############################################################
#### Figure layout
###############################################################
plt.close('all')
fig = plt.figure(figsize = (10, 12.5))
gm = gridspec.GridSpec(460, 600, figure = fig)

ax21 = plt.subplot(gm[:150, :280], projection ="3d")
ax11 = plt.subplot(gm[:150, 320:], projection ="3d")

y0, y1, y2 = 76, 194, 55
axes = [[plt.subplot(gm[y1 + i%2 * y0    : y1+y2 + i%2 *y0,    j%3 *220 : j%3*220 + 160])           for i in range(2)] for j in range(3)]

y0, y1, y2 = 60, 360, 40
axcc = [[plt.subplot(gm[y1 + i%2 * y0    : y1+y2 + i%2 *y0,    j%3 *220 : j%3*220 + 160])      for i in range(2)] for j in range(3)]
###############################################################
#### Plot Neural network CIF distributions
###############################################################
Xph, Y = np.zeros((nfreq, nbins)), np.ones((nfreq, nbins))
for k, f in enumerate(frequencies):
    Xph[k] = xph
    Y[k] *= f
fcut = 0
###############################################################
ax11.plot_surface(np.array(xgmean_control)[fcut:], Y[fcut:] , np.array(CC_ampl_control)[fcut:], cmap=inverted_cmap, vmin = 0, vmax = .6, linewidth=0, antialiased=True)
ax21.plot_surface(np.array(xgmean_sworld)[fcut:], Y[fcut:] , np.array(CC_ampl_sworld)[fcut:], cmap=inverted_cmap, vmin = 0, vmax = .6, linewidth=0, antialiased=True)
for ax in [ax11, ax21]:
    ax.set_xlabel('geometric mean\n (Spk/s)', fontsize = fz)
    ax.set_ylabel('Stimulation frequency (Hz)', fontsize = fz)
    ax.set_zlabel('CIF amplitude', fontsize = fz)
    ax.set_zlim(0, 1.09)
#######################
for ax in [ax11, ax21]:
    ax.view_init(elev=35, azim= -10)  # Change these values as needed
###############################################################
################################################################
for k, freq in enumerate([8, 35, 80]):
    axes[k][0].text(0.5, 1.18, 'Stimulus frequency\n%d Hz'%(freq),
    ha='center', va='center', transform=axes[k][0].transAxes, fontsize = 2+fz)
################################################################
plt.figtext(0.18, 0.98, 'Connected network', ha='left', va='center', fontsize = 18)
plt.figtext(0.74, 0.975, 'Unconnected network\nwith barrage', ha='center', va='center', fontsize = 18)
################################################################
N = 1000
ttot = 1000
twindow, tbin = 1.0, 0.001
timecc = 1000*np.arange(-twindow, twindow+tbin, tbin)

###############################################################
#### Plotting CIF examples at 3 stimulation frequencies
###############################################################
datbins = 200
def pdens_crosscorrelation(spk1, spk2, freq):
    ''' Get the predicted crosscorrelation from spike phase densities '''
    ## phase densities histograms
    h1dat = np.histogram((spk1*freq)%1, range = [0, 1], bins = datbins, density=True)[0]
    h2dat = np.histogram((spk2*freq)%1, range = [0, 1], bins = datbins, density=True)[0]
    ### Calculate ccorrelations from spike phases density
    this_crosscorrelation = np.convolve(np.concatenate([h2dat, h2dat, h2dat]), h1dat[::-1], 'valid')/np.sum(h1dat)
    x = np.linspace(-1000./freq, 1000./freq, 2*datbins +1 )
    return x, this_crosscorrelation
###############################################################

def plot_cc(ax, freq, pair):
    spikes = np.load('./data/spikes_control_%d.npy'%(freq), allow_pickle = True)
    spk1, spk2 = spikes[pair[0]], spikes[pair[1]]
    x, cc1pd = pdens_crosscorrelation(spk1, spk2, freq)
    ax.plot(x, cc1pd, 'k', label = 'Unconnected')

    spikes = np.load('./data/spikes_barrage_%d.npy'%(freq), allow_pickle = True)
    spk1, spk2 = spikes[pair[0]], spikes[pair[1]]
    x, cc2pd = pdens_crosscorrelation(spk1, spk2, freq)
    ax.plot(x, cc2pd, 'r', label = 'Unconnected Barrage')

    spikes = np.load('./data/spikes_sworld_%d.npy'%(freq), allow_pickle = True)
    spk1, spk2 = spikes[pair[0]], spikes[pair[1]]
    x, cc3pd = pdens_crosscorrelation(spk1, spk2, freq)
    ax.plot(x, cc3pd, 'g', label = 'Connected')
    ax.set_xlim(-1000./freq, 1000./freq)
    return 0

hbins = 100
###############################################################
def plot_ang_hist(ax, dat, color, label):
    h, x = np.histogram(dat, range = [0, 1], bins = hbins, density=True)
    xdat = (x[1:] + x[:-1])/2
    ax.plot(xdat, h, '-', lw = 1.6, color = color, label = label)
    ax.fill_between(xdat, h, color = color, alpha = 0.12)
###############################################################
def plot_amp_hist(ax, dat, color, label):
    h, x = np.histogram(dat, range = [0, 1.5], bins = hbins, density=True)
    xdat = (x[1:] + x[:-1])/2
    ax.plot(xdat, h, '-', lw = 1.6, color = color, label = label)
    ax.fill_between(xdat, h, color = color, alpha = 0.12)
###############################################################
###############################################################
frequencies = [8, 35, 80]
colors = ['k', 'r', 'g']
labels = ['Unconnected', 'Unconnected Barrage', 'Connected']

pairs = [[413, 564],  [72, 190]] ## pairs to be plotted as examples in rows E1, E2
################################################################
for k, freq in enumerate(frequencies):
    ### Data contain rate_cell1, rate_cell_2, geom_mean_rate, CC_amplitude, CC_angle
    sw = np.load('./data/analysis_sworld_%d.npy'%(freq), allow_pickle = True)
    co = np.load('./data/analysis_control_%d.npy'%(freq), allow_pickle = True)
    ba = np.load('./data/analysis_barrage_%d.npy'%(freq), allow_pickle = True)
    for j, dat in enumerate([co, ba, sw]):
        plot_amp_hist(axes[k][0], dat[3], colors[j], label = labels[j])
        plot_ang_hist(axes[k][1], dat[4], colors[j], label = labels[j])
    for i, [p1, p2] in enumerate(pairs):
        plot_cc(axcc[k][i], freq, [p1, p2])

axes[0][0].legend(bbox_to_anchor=(.15, 0.93), frameon = False, fontsize = 13)
################################################################
###############################################################
for ax in np.concatenate(axes):
    ax.set_ylabel('PDF', fontsize = fz)

for ax in axes:
    ax[0].set_xlim(-0.01, 1.51)
    ax[0].set_xticks([0, 0.5, 1, 1.5])
    ax[0].set_xticklabels(['0', '0.5', '1', '1.5'])
    ax[0].set_ylim(bottom = -0.01)
    ax[0].set_xlabel('CIF amplitude', fontsize = fz)

for ax in axes:
    ax[1].set_xlim(-0.01, 1.01)
    ax[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax[1].set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax[1].set_ylim(bottom = -0.01)
    ax[1].set_xlabel('CIF phase', fontsize = fz)
#
for ax in np.concatenate(axcc):
    ax.set_ylabel('CIF', fontsize = fz)
    ax.set_xlabel('Time (ms)', fontsize = fz)
    ax.axvline(x = 0, color = 'k', linestyle = '--', linewidth = 0.6)

for ax in np.concatenate(axes + axcc):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
#
# ###############################################################
x0, x1, x2 = 0.1, 0.01, 0.55
y1, y2, y3, y4, y5, fz = 0.94, 0.6, 0.44, 0.27, 0.15,  22
plt.figtext(x0, y1, 'A', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x2, y1, 'B', ha = 'left', va = 'center', fontsize = fz)

plt.figtext(x1, y2, 'C', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x1, y3, 'D', ha = 'left', va = 'center', fontsize = fz)

plt.figtext(x1, y4, 'E1', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x1, y5, 'E2', ha = 'left', va = 'center', fontsize = fz)

fig.subplots_adjust(left = 0.075, bottom = 0.05, right = 0.97, top = 0.980)
plt.savefig('./figure_7.png', dpi = 300)
