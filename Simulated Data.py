import sys
sys.path.append('G:/PHD/Data Analysis Paper/Clustering_Protocol.py')
from Clustering_Protocol import SMS_clustering_protocol
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
#%%

states = {
    "S0": {"mean_intensity": 500, "intensity_sd": 50, "lifetime_mean": 0.8, "lifetime_sd": 0.2}, #1.2
    "S1": {"mean_intensity": 2500, "intensity_sd": 150, "lifetime_mean": 2.2, "lifetime_sd": 0.2}, #3
    "S2": {"mean_intensity": 1200, "intensity_sd": 120, "lifetime_mean": 2.5, "lifetime_sd": 0.2},
    "S3": {"mean_intensity": 1800, "intensity_sd": 180, "lifetime_mean": 1.2, "lifetime_sd": 0.2}
}

def sample_powerlaw(alpha, min_val, max_val):
    r = np.random.uniform(0, 1)
    return ((max_val**(1 - alpha) - min_val**(1 - alpha)) * r + min_val**(1 - alpha))**(1 / (1 - alpha))

def sample_truncated_powerlaw(alpha, tau_c, min_val, max_val):
    while True:
        tau = sample_powerlaw(alpha, min_val, max_val)
        p = np.exp(-tau / tau_c)
        if np.random.uniform(0, 1) < p:
            return tau

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
        p_exc = (np.sin(theta)**2) / 2
        scale *= p_exc

    if apply_drift:
        drift_start = 1.0
        drift_end = 0.5
        drift_factor = drift_start + (drift_end - drift_start) * (time_in_trace / max_trace_length)
        scale *= drift_factor

    return intensity * scale, p_exc


def simulate_particle_trace(particle_id, noise_type=None, short_dwell_threshold=3,
                            affect_this_particle=False,
                            polarization_issue=False, focal_drift=False,
                            ellipticity=1.0, sigma_z=0.7):
    current_state = np.random.choice(["S0", "S1"], p=[0.5, 0.5])
    current_time = 0
    data = []

    # Sample orientation ONCE per particle
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)

    while current_time < max_trace_length:
        if current_state == "S0":
            dwell = sample_powerlaw(m_off, min_dwell, max_dwell)
        else:
            dwell = sample_truncated_powerlaw(m_on, tau_c_on, min_dwell, max_dwell)

        dwell = int(min(dwell, max_trace_length - current_time))
        state = states[current_state]

        raw_intensity = np.random.normal(state["mean_intensity"], state["intensity_sd"])
        p_exc = np.nan

        if current_state == "S1" and affect_this_particle:
            raw_intensity, p_exc = apply_polarization_and_drift(
                raw_intensity,
                theta, phi,
                time_in_trace=current_time,
                max_trace_length=max_trace_length,
                ellipticity=ellipticity,
                apply_polarization=polarization_issue,
                apply_drift=focal_drift,
                sigma_z=sigma_z
            )
            raw_intensity = max(raw_intensity, 500)

        if noise_type:
            intensity = add_noise_to_intensity(raw_intensity, noise_type)
        else:
            intensity = raw_intensity

        lifetime = np.random.normal(state["lifetime_mean"], state["lifetime_sd"])
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
        current_state = np.random.choice(["S0", "S1"], p=[0.5, 0.5])

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
focal_drift = F
polarization_issue = None#'elliptical'
ellipticity = 1
sigma_z = 0

# Choose ~30% of particles to be affected
np.random.seed(42)
affected_particles = set(np.random.choice(range(1, n_particles + 1), size=int(1 * n_particles), replace=False))

# Run simulation
all_data = []
for particle in range(1, n_particles + 1):
    is_affected = particle in affected_particles
    all_data.extend(simulate_particle_trace(
        particle_id=particle,
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
#%%%%%

########################

#testing the simulated dataset with the Clustering protocol.
#SEE BELOW

########################
#%%
combined_Data_group = df
#%%
S1s = combined_Data_group[combined_Data_group['Level'] == "S1"]
plt.hist(S1s['int'])
plt.show()
#%%  
group_df_instance = SMS_clustering_protocol(combined_Data_group)
#%%
group_plot = group_df_instance.contour_plot(levels = 130, xminn=0, yminn=0, ylim=4)
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
#%%
group_int_trace = group_df_instance.int_trace_plot(6, 60, xmin=0, ymin=0, xlim = 600, ylim=2.5)
#%%
group_find_clusters = group_df_instance.find_nr_of_clusters(10, numaxlabl=7,xlim=10.1,ylim=1.03, xmin=0,ymin=-0.05) 

#%%
group_do_clustering = group_df_instance.clustering_the_data(2, xlim=4,ylim=4, conf = 0.9)

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





