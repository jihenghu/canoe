#! /usr/bin/env python3
import numpy as np
import emcee
import sys, os
import matplotlib.pyplot as plt
import h5py
from scipy.stats import norm

from multiprocessing import Pool, current_process
import threading
import queue
import time

sys.path.append("../python")
sys.path.append(".")
from canoe import def_species, load_configure
from canoe.snap import def_thermo
from canoe.athena import Mesh, ParameterInput, Outputs, MeshBlock
# from canoe.harp import radiation_band, radiation

os.environ["OMP_NUM_THREADS"] = "1"
local_storage = threading.local()

def set_atmos_run_RT_concurrent(qNH3: float, 
                     T0: float = 180.0, 
                     RHmax: float=1.0,
                     adlnNH3dlnP: float=0.0,
                     pmin: float = 0.0, 
                     pmax: float = 0.0,
                     jindex: int = 0
                     ):  
    ## construct atmos with a rh limit
    ## jindex is the index of current processer, starting from zero, will add to mb.jst in canoe backend 
    mb.construct_atmosphere(pin, qNH3, T0, RHmax, jindex)

    ## modify the top humidity with a increment
    mb.modify_dlnNH3dlnP_rhmax(adlnNH3dlnP, pmin, pmax, RHmax, jindex) 

    ## do radiative transfer
    rad.cal_radiance(mb, mb.k_st, mb.j_st+jindex)
    tb = np.array([0.0] * 4 * nb)
    for ib in range(nb):
        toa = rad.get_band(ib).get_toa()[0]
        tb[ib * 4 : ib * 4 + 4] = toa
    return tb[4::4]

# Define likelihood function
def ln_likelihood(theta):
    nh3, temp, RHmax, adlnNH3, pmax = theta

    thread_id = current_process().name.split('-')[1] 
    jindex=int(thread_id)-1
    # process_id = os.getpid() 
    # print(f"grid {jindex} is being executed by process ID: {process_id} and thread ID: {thread_id}")

    simulations = set_atmos_run_RT_concurrent(nh3, temp, RHmax, adlnNH3, 1.0E-3,pmax, jindex) 
    # print(simulations)
    residuals =simulations- ref_cpc
    chi_squared = np.sum((residuals / sigma) ** 2)
    return -0.5 * chi_squared

# Define priors for NH3 and temperature
def ln_prior(theta):
    nh3, temp, RHmax, adlnNH3, pmax = theta

    nh3_mean = 380  # Mean value for NH3
    nh3_stddev = 100  # Standard deviation for NH3

    temp_mean = 177.6  # Mean value for temperature
    temp_stddev = 10  # Standard deviation for temperature   0.5%

    RHmax_mean = 0.63  
    RHmax_stddev = 0.5    

    adlnNH3_mean=-0.05
    adlnNH3_stddev=0.8  ## dln100ppmv/ln1.E5

    pmax_mean=4.85E5   ## effective contributing layer of CH4 and CH5
    pmax_stddev=2.0E5

    ln_prior_nh3 = -0.5 * ((nh3 - nh3_mean) / nh3_stddev) ** 2 - np.log(nh3_stddev * np.sqrt(2 * np.pi))
    ln_prior_temp = -0.5 * ((temp - temp_mean) / temp_stddev) ** 2 - np.log(temp_stddev * np.sqrt(2 * np.pi))
    ln_prior_rhmax = -0.5 * ((RHmax - RHmax_mean) / RHmax_stddev) ** 2 - np.log(RHmax_stddev * np.sqrt(2 * np.pi)+ np.log(2))
    ln_prior_adlnNH3 = -0.5 * ((adlnNH3 - adlnNH3_mean) / adlnNH3_stddev) ** 2 - np.log(adlnNH3_stddev * np.sqrt(2 * np.pi)+ np.log(2))
    ln_prior_pmax = -0.5 * ((pmax - pmax_mean) / pmax_stddev) ** 2 - np.log(pmax_stddev * np.sqrt(2 * np.pi)+ np.log(2))

    if (0 < nh3 < 1000) and (100 < temp < 200) and (0 <= RHmax <= 1)  and (5.E4 <= pmax <= 1.E6):
        return ln_prior_nh3 + ln_prior_temp+ln_prior_rhmax+ln_prior_adlnNH3+ln_prior_pmax #
    return -np.inf  # return negative infinity if parameters are outside allowed range

# Combine likelihood and prior to get posterior
def ln_posterior(theta):
    prior = ln_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    return prior + ln_likelihood(theta)

## main
if __name__ == "__main__":

    nx2 = 12  ## shall not be less than N_walkers, can be a little greater for safty.

    ## initialize Canoe
    global pin
    pin = ParameterInput()
    pin.load_from_file("juno_mwr.inp")

    vapors = pin.get_string("species", "vapor").split(", ")
    clouds = pin.get_string("species", "cloud").split(", ")
    tracers = pin.get_string("species", "tracer").split(", ")

    def_species(vapors=vapors, clouds=clouds, tracers=tracers)
    def_thermo(pin)

    config = load_configure("juno_mwr.yaml")
    # print(pin.get_real("problem", "qH2O.ppmv"))

    pin.set_boolean("job","verbose", False)

    print(pin.get_string("mesh","nx2"))
    pin.set_string("mesh","nx2", f"{nx2}")

    print(pin.get_string("mesh","nx2"))

    mesh = Mesh(pin)
    mesh.initialize(pin)

    global mb, rad, nb
    mb = mesh.meshblock(0)
    rad = mb.get_rad()
    nb = rad.get_num_bands()

    # extract multiPJ mean cpcs anomaly
    pvfile= h5py.File("/data/jihenghu/juno-mwr-deconv-research/4.zz.tb/anormaly.corssPJs.h5", "r")
    anomaly = np.array(pvfile["mean_pv"][:,1:])

    anoamly_spectra=np.mean(anomaly, axis=0)

    ## calculate background radiances using posterior results
    tb_bg=set_atmos_run_RT_concurrent(380.38, # NH3.ppmv
                     177.58,       # Temperature
                     0.63,         # RH_max_NH3
                     -0.05,        # adlnNH3/dlnP
                     1E-3,         # pmin [Pa]
                     4.85E5,          # pmax [Pa]
                     0)

    anoamly_spectra[0]=0.0
    print(anoamly_spectra) # [0.        0.7181285 2.1207075 2.969638  1.1837736]
    
    global ref_cpc
    ref_cpc=tb_bg+anoamly_spectra
    print(ref_cpc)  # [479.85371429 333.73846909 253.42675499 196.48110579 142.22219136]

    ##  run MCMC
    ## the signal differences between cpcs and background should be penalized by random noise of 0.5 K, only.
    ##  random error 0.5 K 
    global sigma
    sigma=0.5 #K

    # Initialize walkers
    n_walkers = nx2
    n_dimensions = 5  # nh3, temperature, rh_max_NH3, adlnnh3, pmax
    initial_guess = [200.0, 150.0, 0.5, 0.0, 5.0E5]  # Initial guess for NH3 and temperature
  
    # initial_guesses = [
    #     [initial_guess[i] + initial_guess[i] *0.3* np.random.randn() for i in range(n_dimensions)] for _ in range(n_walkers)
    # ]

    initial_guesses = [
        [730, 120.0, 0.5, 0.0, 5.0E5],
        [150, 195.0, 0.6, -0.1, 2.0E5],
        [500, 155.0, 0.7, -0.2, 3.0E5],
        [250, 132.0, 0.8, -0.3, 2.0E5],
        [320, 165.0, 0.9, -0.5, 2.5E5],
        [100, 170.0, 0.99, 0.01, 3.2E5],
        [820, 140.0, 0.72, -0.05, 5.0E5],
        [980, 130.0, 0.3, -0.21, 4.0E5],
        [610, 112.0, 0.45, -0.15, 3.6E5],
        [405, 182.0, 0.58, -0.1, 2.5E5],
        [385, 199.0, 0.85, 0.0, 3.0E5],
        [590, 145.0, 0.95, 0.6, 5.0E5],
    ]

    # Run MCMC
    n_steps = 5000

    # backend
    filename = f"run_juno_emcee_cpcs_parallel_{n_steps}.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(n_walkers, n_dimensions)

    POOL_SIZE=n_walkers
    with Pool(POOL_SIZE) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dimensions, ln_posterior, backend=backend, pool=pool)
        sampler.run_mcmc(initial_guesses, n_steps, progress=True)

