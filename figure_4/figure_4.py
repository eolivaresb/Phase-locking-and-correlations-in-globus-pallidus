###############################################################
################################################################
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.rcParams.update({'text.usetex': False, 'font.family': 'stixgeneral', 'mathtext.fontset': 'stix',})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
import pickle
pi = np.pi
###############################################################
fz = 16
fz_text = 13.5
################################################################
### Load data
###############################################################
###############################################################
cpdens = ['#ff7f00', '#377eb8']
ccc = ['#4daf4a', '#984ea3']

# Load pdens cc dictionary from file
with open('data/pdens_crosscorrelations.pkl', 'rb') as f:
    pdens_cc = pickle.load(f)

datbins = 30 ## number of bins used in phase density histogram calculation
xpdens = np.arange(0.5/datbins, 1, 1/datbins)
phase_densities = np.load('data/phase_densities.npy', allow_pickle = True)

def plot_pdens(ax, findx, p):
    pd1 = phase_densities[p[1]][findx]
    pd2 = phase_densities[p[2]][findx]
    ax.bar(xpdens, pd1, width = 1/datbins, color = cpdens[0], alpha = 0.4)
    ax.bar(xpdens, pd2, width = 1/datbins, color = cpdens[1], alpha = 0.4)
    ax.text(0.04, 0.97, 'Stim = %d Hz\nCells : {%d, %d}'%(freq, p[1], p[2]), fontsize = fz_text, ha = 'left', transform=ax.transAxes)

def plot_cc(ax, freq, p):
    cc_st = pdens_cc[int(freq-1)][(p[1], p[2])]['spks']
    cc_pd = pdens_cc[int(freq-1)][(p[1], p[2])]['pdens']
    angle = np.abs(pdens_cc[int(freq-1)][(p[1], p[2])]['ang']) / pi
    xdat = 1000*np.linspace(-1./freq, 1./freq, 1+2*datbins)
    ax.plot(xdat, cc_st, color = ccc[0], label = 'Strain CIF')
    ax.plot(xdat, cc_pd, color = ccc[1], label = 'Phase dens CIF')
    ax.text(0.04, 1.01, 'CIF phase = %.2f'%(angle), fontsize = fz_text, ha = 'left', transform=ax.transAxes)
###############################################################
### Stimulation frequency and cell pairs to be plot as examples
beta = [[8, 9, 12], [9, 9, 10], [10, 5, 12], [11, 10, 11]]

lock = [[38, 1, 4], [38, 1, 9], [38, 4, 8], [55, 7, 11]]

high = [[79, 5, 10], [79, 8, 11], [81, 10, 11], [86, 4, 8]]
###############################################################
###############################################################
plt.close('all')
fig = plt.figure(figsize = (14, 12))
gm = gridspec.GridSpec(760, 620, figure = fig)
ax_pd = [[plt.subplot(gm[i%4 * 150 : i%4 *150 + 100, j%3 *220 : j%3*220 + 66]) for i in range(4)] for j in range(3)]
ax_cc = [[plt.subplot(gm[i%4 * 150 : i%4 *150 + 100, j%3 *220 + 90 : j%3*220 + 180]) for i in range(4)] for j in range(3)]

ax1 = plt.subplot(gm[620:760, 10:420])
ax3 = plt.subplot(gm[620:760, 480:])

###############################################################
###############################################################
axp, axc = ax_pd[0], ax_cc[0]
for k, p in enumerate(beta):
    freq, findx = p[0], int(p[0] - 1)

    plot_pdens(axp[k], findx, p)
    plot_cc(axc[k], freq, p)

    axp[k].set_ylim(0, 5.3)
    axc[k].set_ylim(0, 2.3)
###############################################################
###############################################################
axp, axc = ax_pd[1], ax_cc[1]
for k, p in enumerate(lock):
    freq, findx = p[0], int(p[0] - 1)

    plot_pdens(axp[k], findx, p)
    plot_cc(axc[k], freq, p)

    axp[k].set_ylim(0, 7.9)
    axc[k].set_ylim(0, 4.3)
###############################################################
###############################################################
axp, axc = ax_pd[2], ax_cc[2]
for k, p in enumerate(high):
    freq, findx = p[0], int(p[0] - 1)

    plot_pdens(axp[k], findx, p)
    plot_cc(axc[k], freq, p)

    axp[k].set_ylim(0, 5.3)
    axc[k].set_ylim(0, 3.5)
###############################################################
###############################################################
cc_pearson = np.load('data/cc_pearson.npy', allow_pickle = True)
cc_pdens_amp = np.load('data/cc_pdens_amp.npy', allow_pickle = True)
cc_amp = np.load('data/cc_amp.npy', allow_pickle = True)
cc_ang = np.load('data/cc_ang.npy', allow_pickle = True)

mean_dist = np.array([np.mean(p) for p in cc_pearson])
stde_dist = np.array([np.std(p)/np.sqrt(len(p)) for p in cc_pearson])

xfreq = np.arange(1, 101)
ax1.errorbar(xfreq, mean_dist, yerr=stde_dist, fmt='-k')

ax1.set_xlim(-0.5, 100.5)
ax1.set_ylim(0, 1)
ax1.set_xlabel('Stimulation frequency (Hz)', fontsize = fz)
ax1.set_ylabel('Pearson coefficient', fontsize = fz)

for ax in [ax3]: ax.set_xlabel('Amplitude measured\nfrom spike train CIF', fontsize = fz)
ax3.set_ylabel('Amplitude predicted\nfrom phase densities', fontsize = fz)

ax3.plot([0, 1.9], [0, 1.9], '--b')
ms = 0.32
ax3.plot(np.concatenate(cc_amp), np.concatenate(cc_pdens_amp)/2., '.k', ms = ms)
corr_coeff, p_value = pearsonr(np.concatenate(cc_amp), np.concatenate(cc_pdens_amp))
ax3.text(0.15, 1.7, 'r = %.2f'%corr_coeff, fontsize = fz)
ax3.set_xlim(-0.01, 1.8)
ax3.set_ylim(-0.02, 1.8)
###############################################################
###############################################################
for ax in [ax1, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for ax in np.concatenate(ax_pd):
    ax.set_ylabel('PDF', fontsize = fz)
    ax.set_xlim(-0.01, 1.01)

for i in range(3):
    ax_pd[i][-1].set_xlabel('Stimulus phase', fontsize = fz)
    ax_cc[i][-1].set_xlabel('Time (ms)', fontsize = fz)

for ax in np.concatenate(ax_cc):
    ax.set_ylabel('CIF', fontsize = fz)
    ax.axvline(x = 0, color = 'k', linestyle = '--', linewidth = 0.6)

for ax in np.concatenate(ax_pd + ax_cc):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

###############################################################
plt.figtext(0.16, 0.965, 'Low stimulus frequency', ha = 'center', fontsize = 17)
plt.figtext(0.5, 0.965, 'Medium stimulus frequency', ha = 'center', fontsize = 17)
plt.figtext(0.82, 0.965, 'High stimulus frequency', ha = 'center', fontsize = 17)

###############################################################
x1, x2, x3, x4, x5 = 0.01, 0.33, 0.66, 0.07, 0.68
y1, y2, fz = 0.97, 0.24, 24
plt.figtext(x1, y1, 'A', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x2, y1, 'B', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x3, y1, 'C', ha = 'left', va = 'center', fontsize = fz)

plt.figtext(x1, y2, 'D', ha = 'left', va = 'center', fontsize = fz)
plt.figtext(x5, y2, 'E', ha = 'left', va = 'center', fontsize = fz)
# plt.figtext(x5, y2, 'F', ha = 'left', va = 'center', fontsize = fz)

fig.subplots_adjust(left = 0.05, bottom = 0.07, right = 0.97, top = 0.92)
plt.savefig('./figure_4.png', dpi = 300)
