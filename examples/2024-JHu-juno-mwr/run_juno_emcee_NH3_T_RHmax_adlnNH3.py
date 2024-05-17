#! /usr/bin/env python3
import numpy as np
import emcee
import sys, os
import matplotlib.pyplot as plt
import h5py

sys.path.append("../python")
sys.path.append(".")

from canoe import def_species, load_configure
from canoe.snap import def_thermo
from canoe.athena import Mesh, ParameterInput, Outputs, MeshBlock

# from canoe.harp import radiation_band, radiation


# forward operater
def run_RT_modify_atmos(
    mb: MeshBlock, adlnTdlnP: float = 0.0, pmin: float = 0.0, pmax: float = 0.0
) -> np.array:
    # adlnTdlnP=0.0 ## set as insensitive
    mb.modify_dlnTdlnP(adlnTdlnP, pmin, pmax)
    # adlnNH3dlnP = 0#.25
    # mb.modify_dlnNH3dlnP(adlnNH3dlnP, pmin, pmax)

    # for k in range(mb.k_st, mb.k_ed + 1):
    #    for j in range(mb.j_st, mb.j_ed + 1):
    rad = mb.get_rad()
    rad.cal_radiance(mb, mb.k_st, mb.j_st)

    nb = rad.get_num_bands()
    tb = np.array([0.0] * 4 * nb)

    for ib in range(nb):
        print(rad.get_band(ib))
        toa = rad.get_band(ib).get_toa()[0]
        tb[ib * 4 : ib * 4 + 4] = toa
    return tb


def set_atmos_run_RT(qNH3: float, 
                     T0: float = 180.0, 
                     RHmax: float=1.0,
                     adlnNH3dlnP: float=0.0,
                     pmin: float = 0.0, 
                     pmax: float = 0.0
                     ):  
    ## construct atmos with a rh limit
    mb.construct_atmosphere(pin, qNH3, T0, RHmax, 0)

    ## modify the top humidity with a increment
    mb.modify_dlnNH3dlnP_rhmax(adlnNH3dlnP, pmin, pmax, RHmax, 0)

    ## do radiative transfer
    rad = mb.get_rad()
    rad.cal_radiance(mb, mb.k_st, mb.j_st)

    nb = rad.get_num_bands()
    tb = np.array([0.0] * 4 * nb)

    for ib in range(nb):
        toa = rad.get_band(ib).get_toa()[0]
        tb[ib * 4 : ib * 4 + 4] = toa
    return tb[4:]


# Define likelihood function
# def ln_likelihood(theta, observations, observation_errors):
#     nh3, temperature, RHmax = theta

#     simulations = set_atmos_run_RT(nh3, temperature,RHmax)  # Use your forward operator here
#     residuals = observations - simulations
#     print(simulations)
#     print(observations)
#     # print(residuals)
#     chi_squared = np.sum((residuals / observation_errors) ** 2)
#     # print(chi_squared)
#     return -0.5 * chi_squared

# Define likelihood function
def ln_likelihood(theta, observations, COVALL):
    nh3, temp, RHmax, adlnNH3, pmax = theta
    # pmax=pmax*1.E5
    simulations = set_atmos_run_RT(nh3, temp, RHmax, adlnNH3,1.E-3, pmax)  # Use your forward operator here
    residuals = observations - simulations

    chi_squared=0.0
    for ich in range(5): # five channels
        diff=residuals[ich*4:(ich+1)*4]
        cov=COVALL[ich*4:(ich+1)*4,ich*4:(ich+1)*4]
        chi_squared += np.dot(diff, np.linalg.solve(cov, diff))
    return -0.5 * chi_squared


# Define priors for NH3 and temperature
def ln_prior(theta):
    nh3, temp, RHmax, adlnNH3, pmax = theta
    # pmax=pmax*1.E5

    nh3_mean = 300  # Mean value for NH3
    nh3_stddev = 100  # Standard deviation for NH3

    temp_mean = 169  # Mean value for temperature
    temp_stddev = 10  # Standard deviation for temperature   0.5%

    RHmax_mean = 1.0  
    RHmax_stddev = 0.5    

    adlnNH3_mean=0.
    adlnNH3_stddev=0.8  ## dln100ppmv/ln1.E5

    pmax_mean=5.0E5   ## effective contributing layer of CH4 and CH5
    pmax_stddev=1.0E5

    ln_prior_nh3 = -0.5 * ((nh3 - nh3_mean) / nh3_stddev) ** 2 - np.log(
        nh3_stddev * np.sqrt(2 * np.pi)
    )
    ln_prior_temp = -0.5 * ((temp - temp_mean) / temp_stddev) ** 2 - np.log(
        temp_stddev * np.sqrt(2 * np.pi)
    )

    ln_prior_rhmax = -0.5 * ((RHmax - RHmax_mean) / RHmax_stddev) ** 2 - np.log(
        RHmax_stddev * np.sqrt(2 * np.pi)+ np.log(2)
    )

    ln_prior_adlnNH3 = -0.5 * ((adlnNH3 - adlnNH3_mean) / adlnNH3_stddev) ** 2 - np.log(
        adlnNH3_stddev * np.sqrt(2 * np.pi)+ np.log(2)
    )

    ln_prior_pmax = -0.5 * ((pmax - pmax_mean) / pmax_stddev) ** 2 - np.log(
        pmax_stddev * np.sqrt(2 * np.pi)+ np.log(2)
    )

    if (0 < nh3 < 1000) and (100 < temp < 200) and (0 <= RHmax <= 1)  and (5.E4 <= pmax <= 1.E6):
        return ln_prior_nh3 + ln_prior_temp+ln_prior_rhmax+ln_prior_adlnNH3+ln_prior_pmax #
    return -np.inf  # return negative infinity if parameters are outside allowed range


# Combine likelihood and prior to get posterior
def ln_posterior(theta, observations, observation_errors):
    prior = ln_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    return prior + ln_likelihood(theta, observations, observation_errors)


## main
if __name__ == "__main__":
    ##  extract TB observations from ZZ fitting results
    observations = np.zeros((20,))
    obs = np.zeros((24,))
    pj = 51
    mu = np.cos(np.array([0.0, 15.0, 30.0, 45.0]) / 180.0 * np.pi)
    print(mu)
    for ch in range(6):
        tb_file = h5py.File(
            f"/nfs/nuke/chengcli/JUNOMWR/zzhang/PJ{pj:02d}_Freq{ch}.h5", "r"
        )
        if ch == 0:
            c0 = tb_file["ModelTypeupdate1_MultiPJ_Mode1/Iter1/c0"][-1]
            c1 = tb_file["ModelTypeupdate1_MultiPJ_Mode1/Iter1/c1"][-1]
            c2 = tb_file["ModelTypeupdate1_MultiPJ_Mode1/Iter1/c2"][-1]
        else:
            c0 = tb_file["ModelTypeupdate1_MultiPJ_Mode3/Iter2/c0"][-1]
            c1 = tb_file["ModelTypeupdate1_MultiPJ_Mode3/Iter2/c1"][-1]
            c2 = tb_file["ModelTypeupdate1_MultiPJ_Mode3/Iter2/c2"][-1]
        tb_file.close()

        # Xr=1.0 ## \mu >0.6
        obs[(ch) * 4 : (ch + 1) * 4] = (
            c0 - c1 * 5.0 * (1 - mu) + c2 / 0.04 * 0.5 * (mu - 0.8) * (1 - mu)
        )
    ## discard CH1
    observations = obs[4:]
    print(observations)

    # [740.51932939 732.39178625 708.02076917 667.58562359 474.58510281
    # 469.42513666 454.00808555 428.59559118 338.13016122 335.65949356
    # 328.01197674 314.60534003 251.9730167  250.46642377 245.71888005
    # 237.15115289 194.47971955 193.67185714 191.10407859 186.40702401
    # 141.18445694 141.06723252 140.59821156 139.46693178]


    ##  random error 0.5 K
    sigma=0.5 #K
    ndim=20
    noise_var = np.zeros((ndim, ndim))
    # Set the diagonal values to 0.5 * 0.5
    np.fill_diagonal(noise_var, sigma**2)

    ## calibration error 2%
    caliberr=0.02 
    calib_var = np.full((ndim, ndim), caliberr**2)
    calib_var[0:4,4:]=0.0
    calib_var[4:,:4]=0.0
    calib_var[4:8,8:]=0.0
    calib_var[8:,4:8]=0.0
    calib_var[8:12,12:]=0.0
    calib_var[12:,8:12]=0.0
    calib_var[12:16,16:]=0.0
    calib_var[16:,12:16]=0.0

    # Calculate the number of elements in the obs array
    num_obs = len(observations)

    # Create a matrix of size obs*obs filled with zeros
    TT = np.zeros((num_obs, num_obs))

    # Fill the matrix with the product of each element in obs
    for i in range(num_obs):
        for j in range(num_obs):
            TT[i][j] = observations[i] * observations[j]

    COV=TT*calib_var+noise_var

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

    mesh = Mesh(pin)
    mesh.initialize(pin)

    global mb
    mb = mesh.meshblock(0)

    ##  run MCMC

    # Generate synthetic errors (replace with your real data)
    # observation_errors_stddev = 0.02 * observations  ## error = 2%

    # Initialize walkers
    n_walkers = 12
    n_dimensions = 5  # nh3, temperature, rh_max_NH3, adlnnh3, pmax
    initial_guess = [200.0, 150.0, 0.5, 0.0, 5.0E5]  # Initial guess for NH3 and temperature
    # initial_guesses = [
    #     [initial_guess[i] + initial_guess[i] *0.3* np.random.randn() for i in range(n_dimensions)] for _ in range(n_walkers)
    # ]
    # initial_guesses = [
    #     [230, 120.0, 0.5, 0.0, 5.0E5],
    #     [350, 195.0, 0.6, -0.1, 2.0E5],
    #     [500, 155.0, 0.7, -0.2, 3.0E5],
    #     [450, 132.0, 0.8, -0.3, 2.0E5],
    #     [320, 165.0, 0.9, -0.5, 2.5E5],
    #     [400, 170.0, 0.99, 0.01, 3.2E5],
    #     [420, 140.0, 0.72, -0.05, 5.0E5],
    #     [380, 130.0, 0.3, -0.21, 4.0E5],
    #     [310, 112.0, 0.45, -0.15, 3.6E5],
    #     [205, 182.0, 0.58, -0.1, 2.5E5],
    #     [285, 199.0, 0.85, 0.0, 3.0E5],
    #     [290, 145.0, 0.95, 0.0, 5.0E5],
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
        [385, 199.0, 0.85, 0.05, 3.0E5],
        [590, 145.0, 0.95, 0.23, 5.0E5],
    ]

    # Run MCMC
    n_steps = 3000

    # backend
    filename = f"run_mcmc_background_{n_steps}.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(n_walkers, n_dimensions)

    # Set up the sampler
    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dimensions,
        ln_posterior,
        args=(observations, COV),
        backend=backend,
    )

    # Run MCMC
    sampler.run_mcmc(initial_guesses, n_steps, progress=True)

    # ## output tbsimulations 
    # ftbs=h5py.File(f"emcee_tbsimu_RHmax_BK.h5","w")
    # ftbs.create_dataset('tb', data=Tbsimu)
    # ftbs.close()
