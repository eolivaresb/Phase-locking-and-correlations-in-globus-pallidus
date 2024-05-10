###############################################################
###
################################################################
import numpy as np
pi = np.pi
import pickle
################################################################
################################################################
frequencies = np.arange(1, 101)
nfreq = len(frequencies)
nbins = 17
###############################################################
#####################################################
def save_data(name, ph_hist, xgmean, CC_ampl, cc_amp, cc_ang):
    data_to_save = {'a1': ph_hist, 'a2': xgmean, 'a3': CC_ampl, 'a4': cc_amp, 'a5': cc_ang}
    with open('%s.pkl'%name, 'wb') as file:
        pickle.dump(data_to_save, file)

def load_data(name):
    with open('%s.pkl'%name, 'rb') as file:
        ldata = pickle.load(file)
    return [ldata['a1'], ldata['a2'], ldata['a3'], ldata['a4'], ldata['a5']]

def gmean(a, b): return np.sqrt(a *b)
###############################################################
### Experimental data
###############################################################

cc_amp = np.load('../figure_4/data/cc_amp.npy', allow_pickle = True)
cc_ang = np.load('../figure_4/data/cc_ang.npy', allow_pickle = True)
cc_pairs = np.load('../figure_4/data/cc_pairs.npy', allow_pickle = True)
cc_ang = np.abs(cc_ang)/pi*1.
###############################################################
########## CC phase histograms per stimulation frequency
ph = np.zeros((nfreq, nbins))
for findx in range(nfreq):
    ph[findx] = np.histogram(cc_ang[findx], range = [0, 1.], bins = nbins, density=True)[0]
############  Pairs geometric mean rate and CC amplitude matrices
###############################################################
rates_lock = [33, 38, 26, 26, 37, 47, 34, 53, 45, 20, 26, 22, 32, 27, 22, 46]
###############################################################
Xseg, CCamp = np.zeros((nfreq, nbins)), np.ones((nfreq, nbins))

for findx in range(nfreq):
    X = []
    pairs = cc_pairs[findx]
    for k, pair in enumerate(pairs):
        X.append(gmean(rates_lock[pair[1]], rates_lock[pair[0]]))
    Y = cc_amp[findx]
    X, Y = np.array(X), np.array(Y)
    sorted = np.argsort(X)
    edges = np.linspace(0, len(X), nbins+1).astype(int)
    for p in range(nbins):
        Xseg[findx, p] = np.mean(X[sorted][edges[p]:edges[p+1]])
        CCamp[findx, p] = np.mean(Y[sorted][edges[p]:edges[p+1]])

cc_amp = np.array([[np.mean(a), np.std(a)] for a in cc_amp]).T
cc_ang = np.array([[np.mean(a), np.std(a)] for a in cc_ang]).T

save_data('data/data_exp', ph, Xseg, CCamp, cc_amp, cc_ang)

###############################################################
###############################################################
### Neural Network data
##############################################################

ph_hist_sworld, xgmean_sworld, CC_ampl_sworld, cc_amp_sworld, cc_ang_sworld = [], [], [], [], []
ph_hist_control, xgmean_control, CC_ampl_control, cc_amp_control, cc_ang_control = [], [], [], [], []
ph_hist_barrage, xgmean_barrage, CC_ampl_barrage, cc_amp_barrage, cc_ang_barrage = [], [], [], [], []
# ################################################################
# ####################################################
for k, freq in enumerate(frequencies):
    r1, r2, gmean, amp, ang = np.load('./data/analysis_sworld_%d.npy'%freq, allow_pickle = True).T
    ang = np.abs(ang)/pi*1.
    ph_hist_sworld.append(np.histogram(np.abs(ang), range = [0, 1.], bins = nbins, density=True)[0])
    #####
    gm_aggregated, amp_aggregated = np.zeros(nbins), np.ones(nbins)
    sorted = np.argsort(gmean)
    edges = np.linspace(0, len(gmean), nbins+1).astype(int)
    for p in range(nbins):
        gm_aggregated[p] = np.mean(gmean[sorted][edges[p]:edges[p+1]])
        amp_aggregated[p] = np.mean(amp[sorted][edges[p]:edges[p+1]])
    xgmean_sworld.append(gm_aggregated)
    CC_ampl_sworld.append(amp_aggregated)
    cc_amp_sworld.append([np.mean(amp), np.std(amp)])
    cc_ang_sworld.append([np.mean(ang), np.std(ang)])
    #####################################################
    #####################################################
    r1, r2, gmean, amp, ang = np.load('./data/analysis_control_%d.npy'%freq, allow_pickle = True).T
    ang = np.abs(ang)/pi*1.
    ph_hist_control.append(np.histogram(np.abs(ang), range = [0, 1.], bins = nbins, density=True)[0])
    #####
    gm_aggregated, amp_aggregated = np.zeros(nbins), np.ones(nbins)
    sorted = np.argsort(gmean)
    edges = np.linspace(0, len(gmean), nbins+1).astype(int)
    for p in range(nbins):
        gm_aggregated[p] = np.mean(gmean[sorted][edges[p]:edges[p+1]])
        amp_aggregated[p] = np.mean(amp[sorted][edges[p]:edges[p+1]])
    xgmean_control.append(gm_aggregated)
    CC_ampl_control.append(amp_aggregated)
    cc_amp_control.append([np.mean(amp), np.std(amp)])
    cc_ang_control.append([np.mean(ang), np.std(ang)])
    #####################################################
    #####################################################
    r1, r2, gmean, amp, ang = np.load('./data/analysis_barrage_%d.npy'%freq, allow_pickle = True).T
    ang = np.abs(ang)/pi*1.
    ph_hist_barrage.append(np.histogram(np.abs(ang), range = [0, 1.], bins = nbins, density=True)[0])
    #####
    gm_aggregated, amp_aggregated = np.zeros(nbins), np.ones(nbins)
    sorted = np.argsort(gmean)
    edges = np.linspace(0, len(gmean), nbins+1).astype(int)
    for p in range(nbins):
        gm_aggregated[p] = np.mean(gmean[sorted][edges[p]:edges[p+1]])
        amp_aggregated[p] = np.mean(amp[sorted][edges[p]:edges[p+1]])
    xgmean_barrage.append(gm_aggregated)
    CC_ampl_barrage.append(amp_aggregated)
    cc_amp_barrage.append([np.mean(amp), np.std(amp)])
    cc_ang_barrage.append([np.mean(ang), np.std(ang)])
    #####################################################

cc_amp_sworld, cc_ang_sworld = np.array(cc_amp_sworld).T, np.array(cc_ang_sworld).T
cc_amp_control, cc_ang_control = np.array(cc_amp_control).T, np.array(cc_ang_control).T
cc_amp_barrage, cc_ang_barrage = np.array(cc_amp_barrage).T, np.array(cc_ang_barrage).T

save_data('data/data_sworld', ph_hist_sworld, xgmean_sworld, CC_ampl_sworld, cc_amp_sworld, cc_ang_sworld)
save_data('data/data_control', ph_hist_control, xgmean_control, CC_ampl_control, cc_amp_control, cc_ang_control)
save_data('data/data_barrage', ph_hist_barrage, xgmean_barrage, CC_ampl_barrage, cc_amp_barrage, cc_ang_barrage)
###############################################################
