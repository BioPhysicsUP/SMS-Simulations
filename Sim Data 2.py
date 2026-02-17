import sys
sys.path.append('G:/PHD/Data Analysis Paper/Clustering_Protocol.py')
from Clustering_Protocol import SMS_clustering_protocol
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import itertools

#%%

#%%

states = {
    "S0": {"mean_intensity": 500,  "intensity_sd": 120, "lifetime_mean": 0.8, "lifetime_sd": 0.22},
    "S1": {"mean_intensity": 2600, "intensity_sd": 220, "lifetime_mean": 2.1, "lifetime_sd": 0.35},

    # Diffuse off-diagonal / diagonal-merging states
    # Upper-left: dimmer but long-lived, broad
    "S2": {"mean_intensity": 1200,  "intensity_sd": 120, "lifetime_mean": 2.4, "lifetime_sd": 0.1*2.4},   #500    0.5

    # Near-diagonal: between S0 and S1, moderate spread
    "S3": {"mean_intensity": 1500, "intensity_sd": 150, "lifetime_mean": 1.4, "lifetime_sd": 0.1*1.4},   #450   0.4
 
    # Bottom-right: brighter but short-lived, broad
    "S4": {"mean_intensity": 2200, "intensity_sd": 220, "lifetime_mean": 0.8, "lifetime_sd": 0.1*0.8}}   #550   0.4




state_array = ["S0", "S1", "S2", "S3", "S4"]
p_array     = [0.23, 0.23, 0.18, 0.18, 0.18]
p_array     = [0.275, 0.275, 0.15, 0.15, 0.15]
p_array     = [0.35, 0.35, 0.1, 0.1, 0.1]
p_array     = [0.44, 0.44, 0.04, 0.04, 0.04]
#%%
states = {
    "S0": {"mean_intensity": 500, "intensity_sd":  120, "lifetime_mean": 0.8, "lifetime_sd": 0.22/1.5}, #1.2
    "S1": {"mean_intensity": 2600, "intensity_sd": 200, "lifetime_mean": 2.1, "lifetime_sd": 0.3/1.5}}

state_array = ["S0", "S1"]
p_array     = [0.5 , 0.5]

#%%
states = {
    "S0": {"mean_intensity": 500,  "intensity_sd": 120, "lifetime_mean": 0.8, "lifetime_sd": 0.22},
    "S1": {"mean_intensity": 2600, "intensity_sd": 200, "lifetime_mean": 2.1, "lifetime_sd": 0.30},

    # Diffuse off-diagonal / diagonal-merging states
    # Upper-left: dimmer but long-lived, broad
    "S2": {"mean_intensity": 1200,  "intensity_sd": 200, "lifetime_mean": 2.4, "lifetime_sd": 0.2},

    # Near-diagonal: between S0 and S1, moderate spread
    "S3": {"mean_intensity": 1500, "intensity_sd": 200, "lifetime_mean": 1.4, "lifetime_sd": 0.2},

    # Bottom-right: brighter but short-lived, broad
    "S4": {"mean_intensity": 2200, "intensity_sd": 200, "lifetime_mean": 0.8, "lifetime_sd": 0.2},
    
    # Near-diagonal: between S0 and S1, moderate spread
    "S5": {"mean_intensity": 1900, "intensity_sd": 600, "lifetime_mean": 1, "lifetime_sd": 0.4},

    # Bottom-right: brighter but short-lived, broad
    "S6": {"mean_intensity": 1300, "intensity_sd": 600, "lifetime_mean": 2, "lifetime_sd": 0.4},

    # Bottom-right: brighter but short-lived, broad
    "S7": {"mean_intensity": 1000, "intensity_sd": 600, "lifetime_mean": 0.6, "lifetime_sd": 0.4},

    # Bottom-right: brighter but short-lived, broad
    "S8": {"mean_intensity": 3000, "intensity_sd": 600, "lifetime_mean": 1.4, "lifetime_sd": 0.4},

    # Bottom-right: brighter but short-lived, broad
    "S9": {"mean_intensity": 2000, "intensity_sd": 600, "lifetime_mean": 2.2, "lifetime_sd": 0.4}
    
}

state_array = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
p_array     = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

#%%
states = {
    "S0": {"mean_intensity": 500, "intensity_sd":  120, "lifetime_mean": 0.8, "lifetime_sd": 0.22}, #1.2
    "S1": {"mean_intensity": 2600, "intensity_sd": 200, "lifetime_mean": 2.1, "lifetime_sd": 0.3}, #3
    "S2": {"mean_intensity": 1200, "intensity_sd": 240, "lifetime_mean": 2.4, "lifetime_sd": 0.2},
    "S3": {"mean_intensity": 3100, "intensity_sd": 200, "lifetime_mean": 1.1, "lifetime_sd": 0.2},
    "S4": {"mean_intensity": 1500, "intensity_sd": 400/0.8, "lifetime_mean": 0.9, "lifetime_sd": 0.2},
    "S5": {"mean_intensity": 600,  "intensity_sd": 150/0.8, "lifetime_mean": 3.5, "lifetime_sd": 0.2},
    "S6": {"mean_intensity": 3400, "intensity_sd": 350/0.8, "lifetime_mean": 3.5, "lifetime_sd": 0.2},
    "S7": {"mean_intensity": 2200, "intensity_sd": 400/0.8, "lifetime_mean": 3.7, "lifetime_sd": 0.2},
}


# states = {
#     "S0": {"mean_intensity": 500,  "intensity_sd": 120,  "lifetime_mean": 0.8, "lifetime_sd": 0.22},
#     "S1": {"mean_intensity": 2600, "intensity_sd": 220,  "lifetime_mean": 2.1, "lifetime_sd": 0.35},
#     "S2": {"mean_intensity": 1200, "intensity_sd": 300,  "lifetime_mean": 2.4, "lifetime_sd": 0.30},
#     "S3": {"mean_intensity": 3100, "intensity_sd": 260,  "lifetime_mean": 1.1, "lifetime_sd": 0.30},
#     "S4": {"mean_intensity": 1500, "intensity_sd": 520,  "lifetime_mean": 0.9, "lifetime_sd": 0.30},
#     "S5": {"mean_intensity": 600,  "intensity_sd": 260,  "lifetime_mean": 3.3, "lifetime_sd": 0.25},
#     "S6": {"mean_intensity": 3250, "intensity_sd": 420,  "lifetime_mean": 3.4, "lifetime_sd": 0.25},
#     "S7": {"mean_intensity": 2050, "intensity_sd": 480,  "lifetime_mean": 3.6, "lifetime_sd": 0.25},
# }

state_array = ["S0", "S1", "S2", "S3"]
p_array     = [0.25 , 0.25, 0.25, 0.25]

#%%

import numpy as np
from itertools import combinations

def effective_overlap(mu1, sigma1, mu2, sigma2):
    """Compute effective overlap between two 2D Gaussians using Mahalanobis-based formula"""
    cov_sum = np.diag([sigma1[0]**2 + sigma2[0]**2, sigma1[1]**2 + sigma2[1]**2])
    diff = np.array(mu1) - np.array(mu2)
    D_M = np.sqrt(diff.T @ np.linalg.inv(cov_sum) @ diff)
    OVL_eff = np.exp(- (D_M**2) / 8)  # scales 0->1
    return OVL_eff

# Compute pairwise overlaps
pairs = list(combinations(states.keys(), 2))
overlaps = []

print("Pairwise effective overlaps:\n")
for s1, s2 in pairs:
    mu1 = [states[s1]["mean_intensity"], states[s1]["lifetime_mean"]]
    sigma1 = [states[s1]["intensity_sd"], states[s1]["lifetime_sd"]]
    mu2 = [states[s2]["mean_intensity"], states[s2]["lifetime_mean"]]
    sigma2 = [states[s2]["intensity_sd"], states[s2]["lifetime_sd"]]
    OVL = effective_overlap(mu1, sigma1, mu2, sigma2)
    overlaps.append(OVL)
    print(f"{s1}-{s2}: OVL_eff = {OVL:.2f}")

# Average overlap
avg_overlap = np.mean(overlaps)
print(f"\nAverage pairwise effective overlap: {avg_overlap:.2f}")
#%% FAST SIMULATION

# Parameters
min_dwell = 0.5
max_dwell = 60
m_off = 1.8
m_on = 3.5
tau_c_on = 20
max_trace_length = 60 * 10
n_particles = 300
noise_type = 'poisson'

# Drift/polarization
focal_drift = False
polarization_issue = False#'circular'
ellipticity = 0.7
sigma_z = 0

# Choose ~30% of particles to be affected
np.random.seed(42)
affected_particles = set(np.random.choice(range(1, n_particles + 1),
                                          size=int(0.3 * n_particles), replace=False))

# Precompute chi2 or constants if needed (optional)

# Vectorized power-law sampling
def sample_powerlaw(alpha, min_val, max_val, size=1):
    r = np.random.uniform(0, 1, size)
    return ((max_val**(1 - alpha) - min_val**(1 - alpha)) * r + min_val**(1 - alpha))**(1 / (1 - alpha))

# Faster truncated power law (still accept-reject but batched)
def sample_truncated_powerlaw(alpha, tau_c, min_val, max_val):
    while True:
        tau = sample_powerlaw(alpha, min_val, max_val)
        if np.random.rand() < np.exp(-tau / tau_c):
            return tau

# Exponential
def sample_exponential(mean_tau):
    return -mean_tau * np.log(1 - np.random.rand())

# Noise
def add_noise_to_intensity(intensity, noise_type):
    if noise_type == 'poisson':
        return np.random.poisson(lam=max(intensity, 0))
    elif noise_type == 'gaussian':
        return intensity + np.random.normal(0, 0.1*intensity)
    elif noise_type == 'combined':
        poisson_part = np.random.poisson(lam=max(intensity, 0))
        return poisson_part + np.random.normal(0, 0.05*poisson_part)
    else:
        return intensity

# Polarization/drift
def apply_polarization_and_drift(intensity, theta, phi, time_in_trace=0,
                                 max_trace_length=600, ellipticity=1.0,
                                 apply_polarization=None, apply_drift=False):
    scale = 1.0
    if apply_polarization is None:
        scale *= 1
    elif apply_polarization == 'linear':
        scale *= np.cos(phi)**2 * np.sin(theta)**2
    elif apply_polarization == 'elliptical':
        scale *= np.sin(theta)**2 * (np.cos(phi)**2 + ellipticity**2 * np.sin(phi)**2)
    elif apply_polarization == 'circular':
        scale *= np.sin(theta)**2
    if apply_drift:
        drift_factor = 1 + (0.5 - 1) * (time_in_trace / max_trace_length)
        scale *= drift_factor
    return intensity * scale

# Main simulation loop
all_data = []

for particle in range(1, n_particles + 1):
    current_state = np.random.choice(state_array, p=p_array)
    current_time = 0
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    is_affected = particle in affected_particles

    while current_time < max_trace_length:
        state = states[current_state]

        # Dwell time
        if current_state == "S0":  # OFF
            dwell = sample_powerlaw(m_off, min_dwell, max_dwell)
        else:  # ON
            dwell = sample_exponential(4) if current_state=="S1" else sample_truncated_powerlaw(m_on, tau_c_on, min_dwell, max_dwell)
        dwell = min(dwell, max_trace_length - current_time)

        # Correlated intensity & lifetime
        sigma_int, sigma_tau = state["intensity_sd"], state["lifetime_sd"]
        rho = 0.65
        cov_matrix = [[sigma_int**2, rho*sigma_int*sigma_tau],
                      [rho*sigma_int*sigma_tau, sigma_tau**2]]
        raw_intensity, lifetime = np.random.multivariate_normal(
            [state["mean_intensity"], state["lifetime_mean"]],
            cov_matrix
        )

        # Apply polarization/drift if affected
        if is_affected and current_state in ["S1","S2","S3","S4"]:
            raw_intensity = apply_polarization_and_drift(
                raw_intensity, theta, phi,
                time_in_trace=current_time,
                max_trace_length=max_trace_length,
                ellipticity=ellipticity,
                apply_polarization=polarization_issue,
                apply_drift=focal_drift
            )
            raw_intensity = max(raw_intensity, 500)

        # Add noise
        intensity = add_noise_to_intensity(raw_intensity, noise_type)

        # Make lifetime positive
        lifetime = max(lifetime, 0)

        all_data.append({
            "particle": int(particle),
            "Level": str(current_state),
            "start": float(current_time),
            "end": float(current_time + dwell),
            "dwell": float(dwell),
            "int": float(intensity),
            "av_tau": float(lifetime),
            "theta": float(theta),
            "phi": float(phi)
                })


        current_time += dwell
        current_state = np.random.choice(state_array, p=p_array)

# Convert to DataFrame
df = pd.DataFrame(all_data)
df['int'] = df['int'] / 1000
df.to_parquet("simulated_two_state_fluorescence.parquet", index=False)
#%%
combined_Data_group = df
group_df_instance = SMS_clustering_protocol(combined_Data_group)
#%%
def int_trace_plot(df, particle_id, time_res, fontsize = 30, numaxlabl = 7, xlim=25, ylim=3, 
                   xmin=0, ymin=0):
    """
    This function accepts particle_id and time_res as an input, and generates the resolved
    intensity trace, as well as the raw intensity trace
    """
    times, intensities = [], []
    particle_df        = df[df["particle"] == particle_id]
    
    for _, row in particle_df.iterrows():
        t = np.arange(row["start"], row["end"], time_res)
        i = np.full_like(t, row["int"], dtype=float)
        times.append(t)
        intensities.append(i)
    
    df     = df
    #x      = df['Bin Time (s)'].iloc[1:]   # column B, from row 2 onwards
    #y      = df['Bin Int (counts/100ms)'].iloc[1:]   # column C, from row 2 onwards
    plt.figure(dpi=1000, figsize=(10, 6)) #make x 18 for longer looking trace #20
    #plt.step(x, y/100, color = 'grey')            
    t_vals, i_vals = np.concatenate(times), np.concatenate(intensities)
    i_vals = i_vals
    
    plt.step(t_vals, i_vals, label = f"Particle {particle_id}", linewidth = 2, color = 'green')
    plt.xlabel("Time (s)",  fontsize=fontsize)
    plt.ylabel("Intensity (kcounts/s)", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)  # X-axis tick labels
    plt.yticks(fontsize=fontsize)  # Y-axis tick labels
    plt.xlim(xmin, xlim)
    plt.ylim(ymin, ylim)
    plt.show()
    
int_trace_plot(combined_Data_group, 9, 60, xmin=0, ymin=0, xlim = 600, ylim=3)
#%%
group_plot = group_df_instance.contour_plot(levels = 130, xminn=0, yminn=0, ylim=3)
#%%

group_find_clusters = group_df_instance.find_nr_of_clusters(10, numaxlabl=7,xlim=10.5,ylim=1.05, xmin=0, ymin = 0.2, conf = 0.95) 



#%%


#%%
group_do_clustering = group_df_instance.clustering_the_data(3, xlim=4, ylim=3, conf = 0.95)
#%%
from sklearn.metrics import adjusted_rand_score



# True labels
y_true = combined_Data_group['Level'].values

# Predicted cluster labels
y_pred = combined_Data_group['cluster'].values

# Compute ARI
ari = adjusted_rand_score(y_true, y_pred)
print("Adjusted Rand Index (ARI):", ari)

#%%

#Otther clustering 

from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

def fit_bayesian_gmm(X, max_components=6, random_state=42):
    """
    Variational Bayesian Gaussian Mixture Model
    Returns model, labels, effective number of components
    """

    bgmm = BayesianGaussianMixture(
    n_components=8,                  # a bit more than expected K
    covariance_type='full',
    weight_concentration_prior_type='dirichlet_distribution',
    weight_concentration_prior=0.1, # slightly stronger prior to shrink unused components
    mean_precision_prior=1e-2,      # stronger prior to stabilize means
    covariance_prior=None,           # could set small reg_covar to be extra safe
    reg_covar=1e-4,
    max_iter=3000,
    tol=1e-5,
    init_params='kmeans',
    n_init=2,
    random_state=42
    )

    labels = bgmm.fit_predict(X)

    # Effective number of components
    eff_k = np.sum(bgmm.weights_ > 0.01)

    return bgmm, labels, eff_k


X = combined_Data_group[['int', 'av_tau']]

#%% 

gmm = GaussianMixture(10, random_state=42, n_init=5, covariance_type='full')
labels = gmm.fit_predict(X)

#%%
bgmm, bgmm_labels, bgmm_k = fit_bayesian_gmm(X)

print(f"Bayesian GMM effective K = {bgmm_k}")


#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
import seaborn as sns
import numpy as np
from matplotlib import cm
from scipy.stats import chi2

def plot_sms_kde_with_bgmm(int_data, tau_data, bgmm, conf=0.95, fontsize=30, 
                           xlbl='Intensity', ylbl='Lifetime', 
                           xminn=None, xlim=None, yminn=None, ylim=3, 
                           numaxlabl=6, levels=20):
    """
    Plot 2D KDE of SMS data with Bayesian GMM cluster centers and covariance ellipses.

    Parameters
    ----------
    int_data : array-like
        Intensity values
    tau_data : array-like
        Lifetime values
    bgmm : fitted BayesianGaussianMixture object
    conf : float
        Confidence interval for ellipses
    Other parameters control plotting aesthetics.
    """
    
    # Filter clusters with meaningful weight (>1%)
    active_idx = np.where(bgmm.weights_ > 0.01)[0]
    cluster_nr = len(active_idx)
    
    plt.figure(dpi=1000, figsize=(10, 6))
    ax = plt.gca()
    
    # KDE background
    sns.kdeplot(
        x=int_data,
        y=tau_data,
        cmap='turbo',
        bw_adjust=0.6,
        fill=True,
        alpha=0.75,
        common_grid=True,
        levels=levels,
        ax=ax
    )
    
    # Colormap for clusters
    colormap = cm.get_cmap('Paired', cluster_nr)
    colors = [colormap(i) for i in range(cluster_nr)]
    
    # Cluster centers
    centers = bgmm.means_[active_idx]
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=1, marker='X')
    
    # Draw ellipses for each active cluster
# Draw ellipses for each active cluster
    for i, idx in enumerate(active_idx):
        mean = bgmm.means_[idx]
        cov = bgmm.covariances_[idx]
        color = colors[i % len(colors)]
        
        
        if cov.shape == (2, 2):
            # Compute eigenvalues and eigenvectors
            vals, vecs = np.linalg.eigh(cov)  # Use eigh for symmetric matrices
            order = vals.argsort()[::-1]  # descending order
            vals = vals[order]
            vecs = vecs[:, order]

            # Angle of ellipse (in degrees)
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

            # Width and height = 2 * sqrt(eigenvalues) scaled by chi2
            width, height = 2 * np.sqrt(vals * chi2.ppf(conf, df=2))
        else:
            angle = 0
            width, height = 2 * np.sqrt(cov * chi2.ppf(conf, df=2))
        
        ell = Ellipse(
                xy=mean,
                width=width,
                height=height,
                angle=angle,
                edgecolor=color,
                facecolor='none',
                lw=2,
                alpha=1
                )
        
        ax.add_patch(ell)


    
    # Colorbar for KDE
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap='turbo', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.ax.tick_params(labelsize=fontsize)
    
    # Background
    ax.patch.set_facecolor('#36215e')
    
    # Labels and axis limits
    plt.xlabel(xlbl, fontsize=fontsize)
    plt.ylabel(ylbl, fontsize=fontsize)
    if xminn is not None and xlim is not None:
        plt.xlim(xminn, xlim)
    if yminn is not None and ylim is not None:
        plt.ylim(yminn, ylim)
    plt.locator_params(axis='x', nbins=numaxlabl)
    plt.locator_params(axis='y', nbins=numaxlabl)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    # Sort and label centers by intensity
    int_centres = centers[:, 0]
    tau_centers = centers[:, 1]
    sorted_centers = sorted(zip(int_centres, tau_centers))
    int_cent_sort, tau_cent_sort = zip(*sorted_centers)
    for i, (x, y) in enumerate(zip(int_cent_sort, tau_cent_sort)):
        plt.text(x + 0.05, y + 0.05, f"{i+1}", color='black', fontsize=fontsize, fontweight='bold')
    
    plt.xlabel('Intensity (kcounts/s)', fontsize=fontsize)
    plt.ylabel('Lifetime (ns)', fontsize=fontsize)
    plt.xlim(xminn, xlim)
    plt.ylim(yminn, ylim)
    plt.locator_params(axis='x', nbins=numaxlabl)
    plt.locator_params(axis='y', nbins=numaxlabl)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.show()



#%%

#%%
# Extract intensity and lifetime from your DataFrame
int_data = combined_Data_group['int'].values
tau_data = combined_Data_group['av_tau'].values

# Plot KDE with Bayesian GMM cluster centers
plot_sms_kde_with_bgmm(int_data, tau_data, bgmm,
                       conf=0.95,
                       fontsize=30,
                       xlbl='Intensity (a.u.)',
                       ylbl='Lifetime (ns)',
                       levels=20)








#%%



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Your data
X = combined_Data_group[['int', 'av_tau']]

# Range of K to test
K_range = range(1, 8)

inertia = []
silhouette = []

for k in K_range:
    km = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=42
    )
    labels = km.fit_predict(X)
    inertia.append(km.inertia_)

    if k > 1:
        silhouette.append(silhouette_score(X, labels))
    else:
        silhouette.append(np.nan)



#%%

fontsize = 30
plt.figure(figsize=(10, 6), dpi=1000)
plt.plot(K_range, inertia/np.max(inertia), 'o--', color = 'red', markersize=8)
plt.xlabel("Number of clusters K", fontsize = fontsize)
plt.ylabel("Normalized WCSS", fontsize = fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()
plt.show()




#%%




plt.figure(figsize=(6, 4), dpi=300)
plt.plot(K_range, silhouette, 'o-')
plt.xlabel("Number of clusters K")
plt.ylabel("Mean silhouette score")
plt.title("K-means Silhouette Criterion")
plt.tight_layout()
plt.show()



#%%




X = combined_Data_group[['int', 'av_tau']].values

k_opt = 5   # or whatever elbow/silhouette suggests

kmeans = KMeans(
    n_clusters=k_opt,
    n_init=20,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_



#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def plot_sms_kde_with_kmeans_boundaries(int_data, tau_data, kmeans,
                                        fontsize=30,
                                        xlbl='Intensity (kcounts/s)', ylbl='Lifetime (ns)',
                                        xminn=None, xlim=None, yminn=None, ylim=None,
                                        numaxlabl=7, levels=20,
                                        grid_res=300):
    """
    Plot 2D KDE of SMS data with K-means Voronoi boundaries and centers.
    """

    X = np.column_stack([int_data, tau_data])
    centers = kmeans.cluster_centers_

    # Axis limits
    xmin = xminn if xminn is not None else X[:, 0].min()
    xmax = xlim  if xlim  is not None else X[:, 0].max()
    ymin = yminn if yminn is not None else X[:, 1].min()
    ymax = ylim  if ylim  is not None else X[:, 1].max()

    # Grid for decision regions
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, grid_res),
        np.linspace(ymin, ymax, grid_res)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = kmeans.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(dpi=1000, figsize=(10, 6))
    ax = plt.gca()

    # Plot decision regions
    ax.contourf(
        xx, yy, Z,
        alpha=0.25,
        cmap='tab10',
        antialiased=True
    )

    # KDE background
    sns.kdeplot(
        x=int_data,
        y=tau_data,
        cmap='turbo',
        bw_adjust=0.6,
        fill=True,
        alpha=0.75,
        levels=levels,
        ax=ax
    )

    # Cluster centers
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c='black',
        s=200,
        marker='X',
        zorder=5
    )
    
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap='turbo', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.ax.tick_params(labelsize=fontsize)
    
    for i, (x, y) in enumerate(centers): 
        plt.text(x + 0.05, y + 0.05, f"{i+1}", color='black', fontsize=fontsize, 
                 fontweight='bold')

    # Labels
    plt.xlabel(xlbl, fontsize=fontsize)
    plt.ylabel(ylbl, fontsize=fontsize)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.locator_params(axis='x', nbins=numaxlabl)
    plt.locator_params(axis='y', nbins=numaxlabl)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ax.patch.set_facecolor('#36215e')
    plt.tight_layout()
    plt.show()

    
    
#%%

X = combined_Data_group[['int', 'av_tau']].values

plot_sms_kde_with_kmeans_boundaries(
    int_data=X[:, 0],
    tau_data=X[:, 1],
    kmeans=kmeans,
    xminn=0,
    xlim=4,
    yminn=0,
    ylim=3
)




