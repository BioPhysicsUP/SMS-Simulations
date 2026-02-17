# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 11:52:52 2025

@author: Mikey
"""
import sys
sys.path.append('G:/PHD/Data Analysis Paper/Clustering_Protocol.py')
from Clustering_Protocol import SMS_clustering_protocol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

#%%

# -----------------------------
# Simulation parameters
# -----------------------------
np.random.seed(42)

n_particles = 120
states = {
    0: {"peak": 610, "fwhm": 18},
    1: {"peak": 625, "fwhm": 30},
    2: {"peak": 645, "fwhm": 50},
    3: {"peak": 670, "fwhm": 85},  # diffuse state
}

state_probs = [0.35, 0.30, 0.25, 0.10]

peak_noise = [1.5, 2.5, 4.0, 8.0]
fwhm_noise = [2.0, 3.5, 6.0, 12.0]

# -----------------------------
# Generate data
# -----------------------------
rows = []

for pid in range(n_particles):
    s = np.random.choice(list(states.keys()), p=state_probs)

    peak = (
        states[s]["peak"]
        + 0.35 * (states[s]["fwhm"] - 30)  # correlation term
        + np.random.normal(0, peak_noise[s])
    )

    fwhm = states[s]["fwhm"] + np.random.normal(0, fwhm_noise[s])

    rows.append({
        "particle": pid,
        "state_true": s,
        "peak_nm": peak,
        "fwhm_nm": fwhm
    })

df = pd.DataFrame(rows)

#%%
fontsize = 30
numaxlabl = 7
plt.figure(dpi=1000, figsize=(10, 6))
plt.scatter(df["peak_nm"], df["fwhm_nm"], s=100, alpha=0.85, color = 'red')
plt.xticks(fontsize=fontsize)  # X-axis tick labels
plt.yticks(fontsize=fontsize)  # Y-axis tick labels
plt.xlabel("Spectral Peak Position (nm)", fontsize=fontsize)
plt.ylabel("FWHM (nm)", fontsize=fontsize)
plt.ylim(0, 120)

plt.show()

#%%

# Wavelength axis
wl = np.linspace(580, 720, 2000)

def gaussian(wl, mu, fwhm, amp=1.0):
    sigma = fwhm / 2.355
    return amp * np.exp(-0.5 * ((wl - mu) / sigma)**2)

# -----------------------------
# Build spectra
# -----------------------------
spectra = []

for _, row in df.iterrows():
    spec = gaussian(
        wl,
        row["peak_nm"],
        row["fwhm_nm"],
        amp=1.0
    )
    spectra.append(spec)

spectra = np.array(spectra)

ensemble = spectra.sum(axis=0)

plt.figure(dpi=1000, figsize=(10, 6))
plt.plot(wl, ensemble, lw=4, color = 'red')
plt.xlabel("Wavelength (nm)", fontsize=fontsize)
plt.ylabel("Intensity (arb. units)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)  # X-axis tick labels
plt.yticks(fontsize=fontsize) 
plt.locator_params(axis='x', nbins=numaxlabl)
plt.locator_params(axis='y', nbins=numaxlabl)

plt.show()



#%%

df['int']  = df['peak_nm'] 
df['av_tau']  = df['fwhm_nm'] 

combined_Data_group = df


group_df_instance = SMS_clustering_protocol(combined_Data_group)
group_plot = group_df_instance.contour_plot(levels = 130, xminn=590, yminn=0, ylim=120, xlim = 710, xdata = 'peak_nm', ydata="fwhm_nm", xlbl = 'Peak position (nm)', ylbl = 'FWHM (nm)' , numaxlabl=7)
#%%

group_find_clusters = group_df_instance.find_nr_of_clusters(10, numaxlabl=7,xlim=10.5,ylim=1.03, xmin=0, ymin = 0.75, conf = 0.95) 
#%%
group_do_clustering = group_df_instance.clustering_the_data(4,  xminn=580, yminn=0, ylim=120, xlim = 720 , conf = 0.95, xlbl = 'Peak position (nm)', ylbl = 'FWHM (nm)' , numaxlabl=7)

#%%

cluster_centers = group_do_clustering

cluster_cnt_err = group_df_instance.ground_truth_recovery(states, cluster_centers, int_lbl = "peak", tau_lbl = 'fwhm')
cluster_cnt_err

#%%
from scipy.optimize import linear_sum_assignment

def ground_truth_recovery( true_states, gmm_means, int_lbl = "peak", tau_lbl = "fwhm" ):
    """ Both have array shape (n_states, 2) -> [int, tau] """
    
    true_centers = np.array([[v[int_lbl], v[tau_lbl]] 
                         for v in true_states.values()])
    
    # Compute distance matrix
    D = np.linalg.norm(true_centers[:, None, :] - gmm_means[None, :, :], axis=2)

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(D)

    mean_int_error = np.abs(true_centers[row_ind, 0] - gmm_means[col_ind, 0])
    mean_tau_error = np.abs(true_centers[row_ind, 1] - gmm_means[col_ind, 1])
    rmse = np.sqrt(np.mean(D[row_ind, col_ind]**2))

    errors = {
        "mean_int_error": mean_int_error,
        "mean_tau_error": mean_tau_error,
        "rmse": rmse,
        "mapping": (row_ind, col_ind)
        }
    return errors

#%%
ground_truth_recovery( states, cluster_centers, int_lbl = "peak", tau_lbl = "fwhm" )