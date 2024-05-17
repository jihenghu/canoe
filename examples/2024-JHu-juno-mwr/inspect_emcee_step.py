# import emcee

# %config InlineBackend.figure_format = "retina"
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# reader = emcee.backends.HDFBackend("run_mcmc.h5")

# tau = reader.get_autocorr_time()
# burnin = int(2 * np.max(tau))
# thin = int(0.5 * np.min(tau))
# samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
# log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)

import h5py

h5 = h5py.File("run_mcmc_cpcs_noch2_1000.h5", "r")
chain = h5["mcmc"]["chain"][:,:,:]
h5.close()

flattened_chain = chain.reshape(-1, 2)

# Create labels for the parameters
# labels = ["qNH3 [ppmv]", "Temperature [K]", "RHmax"]

# Create the corner plot
fig, ax = plt.subplots(5, 1, figsize=(10, 10))
for iw in range(12):
    ax[0].plot(range(1000), chain[:, iw, 0],label=iw)

ax[0].set_ylabel("qNH3 [ppmv]")
ax[0].set_xlim([0, 1000])
ax[0].legend()

for iw in range(12):
    ax[1].plot(range(1000), chain[:, iw, 1])
ax[1].set_ylabel("Temperature [K]")
ax[1].set_xlim([0, 1000])
ax[1].set_xlabel("step")

for iw in range(12):
    ax[2].plot(range(1000), chain[:, iw, 2])
ax[2].set_ylabel("RH_max_NH3")
ax[2].set_xlim([0, 1000])
ax[2].set_xlabel("step")

for iw in range(12):
    ax[3].plot(range(1000), chain[:, iw, 3])
ax[3].set_ylabel("a_dlnNH3/dlnP")
ax[3].set_xlim([0, 1000])
ax[3].set_xlabel("step")

for iw in range(12):
    ax[4].plot(range(1000), chain[:, iw, 4])
ax[4].set_ylabel("a_Pmax")
ax[4].set_xlim([0, 1000])
ax[4].set_xlabel("step")

plt.tight_layout()
# Show the plot
plt.savefig("run_mcmc_cpcs_noch2_step_1000.png")
