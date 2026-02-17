import sys
sys.path.append('G:/PHD/Data Analysis Paper/Clustering_Protocol.py')
from Clustering_Protocol import SMS_clustering_protocol
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
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

states = {
    "S0": {"mean_intensity": 500, "intensity_sd":  120, "lifetime_mean": 0.8, "lifetime_sd": 0.22/1.5}, #1.2
    "S1": {"mean_intensity": 2600, "intensity_sd": 200, "lifetime_mean": 2.1, "lifetime_sd": 0.3/1.5}}

state_array = ["S0", "S1"]
p_array     = [0.5 , 0.5]
#%%
state_array = ["S0", "S1", "S2", "S3"]
p_array     = [0.25 , 0.25, 0.25, 0.25]
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
    "S0": {"mean_intensity": 500,  "intensity_sd": 120, "lifetime_mean": 0.8, "lifetime_sd": 0.22},
    "S1": {"mean_intensity": 2600, "intensity_sd": 220, "lifetime_mean": 2.1, "lifetime_sd": 0.35},

    # Diffuse off-diagonal / diagonal-merging states
    # Upper-left: dimmer but long-lived, broad
    "S2": {"mean_intensity": 1200,  "intensity_sd": 500, "lifetime_mean": 2.4, "lifetime_sd": 0.5},

    # Near-diagonal: between S0 and S1, moderate spread
    "S3": {"mean_intensity": 1500, "intensity_sd": 450, "lifetime_mean": 1.4, "lifetime_sd": 0.4},

    # Bottom-right: brighter but short-lived, broad
    "S4": {"mean_intensity": 2200, "intensity_sd": 550, "lifetime_mean": 0.8, "lifetime_sd": 0.4}}




state_array = ["S0", "S1", "S2", "S3", "S4"]
p_array     = [0.35, 0.35, 0.1, 0.1, 0.1]
#%%



#%%

def sample_powerlaw(alpha, min_val, max_val):
    r = np.random.uniform(0, 1)
    return ((max_val**(1 - alpha) - min_val**(1 - alpha)) * r + min_val**(1 - alpha))**(1 / (1 - alpha))

def sample_truncated_powerlaw(alpha, tau_c, min_val, max_val):
    while True:
        tau = sample_powerlaw(alpha, min_val, max_val)
        p = np.exp(-tau / tau_c)
        if np.random.uniform(0, 1) < p:
            return tau
        
def sample_exponential(mean_tau):
    r = np.random.uniform(0, 1)
    return -mean_tau * np.log(1 - r)


def add_noise_to_intensity(intensity, noise_type):
    if noise_type == 'poisson':
        return np.random.poisson(lam=max(intensity, 0))
    elif noise_type == 'gaussian':
        return intensity + np.random.normal(0, 0.1 * intensity)
    elif noise_type == 'combined':
        poisson_part = np.random.poisson(lam=max(intensity, 0))
        return poisson_part + np.random.normal(0, 0.05 * poisson_part)
    else:
        return intensity

def apply_polarization_and_drift(intensity, theta, phi, time_in_trace=0, max_trace_length=600,
                                 ellipticity=1.0,
                                 apply_polarization=None, apply_drift=False,
                                 sigma_z=0.5):
    scale = 1.0
    p_exc = np.nan  # default if no polarization applied

    if apply_polarization == None:
        scale *= 1

    elif apply_polarization == 'linear':
        p_exc = np.cos(phi)**2 * np.sin(theta)**2
        scale *= p_exc

    elif apply_polarization == 'elliptical':
        numerator = np.sin(theta)**2 * (np.cos(phi)**2 + ellipticity**2 * np.sin(phi)**2)
        denominator = 1 + ellipticity**2
        p_exc = numerator  # / denominator if you want normalized
        scale *= p_exc

    elif apply_polarization == 'circular':
        p_exc = (np.sin(theta)**2)# / 2
        scale *= p_exc

    if apply_drift:
        drift_start = 1.0
        drift_end = 0.5
        drift_factor = drift_start + (drift_end - drift_start) * (time_in_trace / max_trace_length)
        scale *= drift_factor

    return intensity * scale, p_exc


def simulate_particle_trace(particle_id, particle_type='Dye', noise_type=None, short_dwell_threshold=3,
                            affect_this_particle=False,
                            polarization_issue=False, focal_drift=False,
                            ellipticity=1.0, sigma_z=0.7):
    
    current_state = np.random.choice(state_array, p=p_array)
    current_time = 0
    data = []

    # Sample orientation ONCE per particle
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)

    while current_time < max_trace_length:
        # Determine dwell time based on particle type and state
        if current_state == "S0":  # OFF
            if particle_type == 'QD':
                dwell = sample_powerlaw(m_off, min_dwell, max_dwell)
            elif particle_type == 'Dye':
                dwell = sample_powerlaw(m_off, min_dwell, max_dwell)
            elif particle_type == 'bio':
                dwell = sample_powerlaw(m_off, min_dwell, max_dwell)
            elif particle_type == 'expo_only':
                dwell = sample_exponential(2)
                
        else:  # ON
            if particle_type == 'QD':
                dwell = sample_powerlaw(m_on, min_dwell, max_dwell)
            elif particle_type == 'Dye':
                dwell = sample_exponential(4)
            elif particle_type == 'bio':
                dwell = sample_truncated_powerlaw(m_on, tau_c_on, min_dwell, max_dwell)
            elif particle_type == 'expo_only':
                dwell = sample_exponential(4)

        dwell = int(min(dwell, max_trace_length - current_time))
        state = states[current_state]
        
        # Correlated intensity & lifetime

        sigma_int = state["intensity_sd"]
        sigma_tau = state["lifetime_sd"]
        rho=0.65
        cov_matrix = [[sigma_int**2, rho*sigma_int*sigma_tau],
                      [rho*sigma_int*sigma_tau, sigma_tau**2]]
        raw_intensity, lifetime = np.random.multivariate_normal(
            [state["mean_intensity"], state["lifetime_mean"]], cov_matrix
        )

        # Apply polarization/focal drift if affected
        if current_state == "S1" and affect_this_particle or current_state == "S2" and affect_this_particle or current_state == "S3" and affect_this_particle \
        or current_state == "S4" and affect_this_particle or current_state == "S5" and affect_this_particle or current_state == "S6" and affect_this_particle \
                or current_state == "S7" and affect_this_particle:
            raw_intensity, p_exc = apply_polarization_and_drift(
                raw_intensity, theta, phi,
                time_in_trace=current_time,
                max_trace_length=max_trace_length,
                ellipticity=ellipticity,
                apply_polarization=polarization_issue,
                apply_drift=focal_drift,
                sigma_z=sigma_z
            )
            raw_intensity = max(raw_intensity, 500)#states[current_state]["mean_intensity"]/1.5)
        else:
            p_exc = np.nan

        # Add shot/gaussian noise to intensity
        if noise_type:
            intensity = add_noise_to_intensity(raw_intensity, noise_type)
        else:
            intensity = raw_intensity

        # Make lifetime positive
        lifetime = max(lifetime, 0)


        data.append({
            "particle": particle_id,
            "Level": current_state,
            "start": current_time,
            "end": current_time + dwell,
            "dwell": dwell,
            "int": intensity,
            "av_tau": lifetime,
            "theta": theta,
            "phi": phi,
            "p_exc": p_exc
        })

        current_time += dwell
        current_state = np.random.choice(state_array, p=p_array)#current_state = np.random.choice(["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"], p=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])

    return data



# Simulation parameters
min_dwell = 0.5
max_dwell = 60
m_off = 1.8
m_on = 1.2
tau_c_on = 20
max_trace_length = 60 * 10
n_particles = 300
noise_type = 'poisson'

# Drift and polarization control
T = True
F = False
focal_drift = F#T
polarization_issue = F#'circular'
ellipticity = 0.7
sigma_z = 0

# Choose ~30% of particles to be affected
np.random.seed(42)
affected_particles = set(np.random.choice(range(1, n_particles + 1), size=int(1 * n_particles), replace=False))

# Run simulation
all_data = []
for particle in range(1, n_particles + 1):
    is_affected = particle in affected_particles
    all_data.extend(simulate_particle_trace(
        particle_id=particle,particle_type='bio',
        noise_type=noise_type,
        affect_this_particle=is_affected,
        polarization_issue=polarization_issue,
        focal_drift=focal_drift,
        ellipticity=ellipticity,
        sigma_z=sigma_z
    ))

df = pd.DataFrame(all_data)
df['int'] = df['int'] / 1000
df.to_parquet("simulated_two_state_fluorescence_with_drift_polarization.parquet", index=False)
df.head()

combined_Data_group = df

#%%  
group_df_instance = SMS_clustering_protocol(combined_Data_group)


#%%
group_plot = group_df_instance.contour_plot(levels = 130, xminn=0, yminn=0, ylim=3)
#%% #Find ratio of affected mean in to unaffected  should =statitsical average = 1/3

df['affected'] = df['particle'].isin(affected_particles)


# Focus on S1 states
s1_df = df[df['Level'] == 'S1']

# Compute group means
mean_affected = s1_df[s1_df['affected']]['int'].mean()
mean_unaffected = s1_df[~s1_df['affected']]['int'].mean()

# Print the results
print(f"Mean S1 intensity (unaffected): {mean_unaffected:.2f}")
print(f"Mean S1 intensity (affected):   {mean_affected:.2f}")
print(f"Suppression ratio:              {mean_affected / mean_unaffected:.2f}")
#Mean S1 intensity (unaffected): 2.50
#Mean S1 intensity (affected):   0.83
#Suppression ratio:              0.33
#%% $Find INt trace
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
    
int_trace_plot(combined_Data_group, 9, 60, xmin=0, ymin=0, xlim = 600, ylim=4)
#%%

#%%
group_find_clusters = group_df_instance.find_nr_of_clusters(10, numaxlabl=7,xlim=10.5,ylim=1.05, xmin=0, ymin = 0, conf = 0.95) 



#%%


#%%
group_do_clustering = group_df_instance.clustering_the_data(4, xlim=4, ylim=3, conf = 0.95)
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

cluster_centers = group_do_clustering

cluster_cnt_err = group_df_instance.ground_truth_recovery(states, cluster_centers)
cluster_cnt_err

#%% get cluster centres and do some sorting

df_c = group_df_instance.make_cluster_centre_df()
int_cent_sort, tau_cent_sort = group_df_instance.sort_cluster_centres()
cluster_order = group_df_instance.get_ordered_clusters(int_cent_sort)
get_dwells = group_df_instance.get_dwells_of_sorted_clusters( combined_Data_group, cluster_order)

#%%
lower, upper  = 1, 0

filtered_df1 = group_df_instance.filtered_IQR_data_to_df(1.5, df_c) # filtered based on median + k * IQR
#%%

filtered_instance   = SMS_clustering_protocol(filtered_df1)
filtered_getd_wells = filtered_instance.get_dwells_of_sorted_clusters( filtered_df1, cluster_order)
#%%
filtered_contour = filtered_instance.contour_plot()
ratio = len(filtered_df1[filtered_df1['cluster']==0])/len(combined_Data_group[combined_Data_group['cluster']==0])

#%%
filtered_rates = filtered_instance.rate_freq_dwelltime(filtered_df1, lower, upper)
#%%
filter_weighted       = filtered_instance.weighted_int_and_tau(filtered_df1)
filter_clus_lifetimes = filtered_instance.determining_cluster_lifetimes(filtered_df1, int_cent_sort, cluster_order)


#%% Multi State


#%%  
group_df_instance = SMS_clustering_protocol(combined_Data_group)
group_plot = group_df_instance.contour_plot()
#%%
group_int_trace = group_df_instance.int_trace_plot(55, 60)

#%%
group_find_clusters = group_df_instance.find_nr_of_clusters(20) 

#%%
group_do_clustering = group_df_instance.clustering_the_data(4)

#%% get cluster centres and do some sorting

df_c = group_df_instance.make_cluster_centre_df()
int_cent_sort, tau_cent_sort = group_df_instance.sort_cluster_centres()
cluster_order = group_df_instance.get_ordered_clusters(int_cent_sort)
get_dwells = group_df_instance.get_dwells_of_sorted_clusters( combined_Data_group, cluster_order)

#%%
a, b, c, d  = 1, 3, 0, 2

filtered_df1 = group_df_instance.filtered_IQR_data_to_df(1.5, df_c) # filtered based on median + k * IQR
#%%

filtered_instance   = SMS_clustering_protocol(filtered_df1)
filtered_getd_wells = filtered_instance.get_dwells_of_sorted_clusters( filtered_df1, cluster_order)
#%%
filtered_contour = filtered_instance.contour_plot()
ratio = len(filtered_df1[filtered_df1['cluster']==0])/len(combined_Data_group[combined_Data_group['cluster']==0])
ratio
#%%
filtered_rates = filtered_instance.rate_freq_dwelltime(filtered_df1, a, b)
#%%
filtered_rates = filtered_instance.rate_freq_dwelltime(filtered_df1, a, c)
#%%
filtered_rates = filtered_instance.rate_freq_dwelltime(filtered_df1, a, d)
#%%
filtered_rates = filtered_instance.rate_freq_dwelltime(filtered_df1, b, c)
#%%
filtered_rates = filtered_instance.rate_freq_dwelltime(filtered_df1, b, d)
#%%
filtered_rates = filtered_instance.rate_freq_dwelltime(filtered_df1, c, d)
#%%
filter_weighted       = filtered_instance.weighted_int_and_tau(filtered_df1)
filter_clus_lifetimes = filtered_instance.determining_cluster_lifetimes(filtered_df1, int_cent_sort, cluster_order)





