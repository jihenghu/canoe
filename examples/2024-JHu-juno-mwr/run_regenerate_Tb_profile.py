#! /usr/bin/env python3
import numpy as np
import emcee
import sys, os
import matplotlib.pyplot as plt
import h5py

sys.path.append("../python")
sys.path.append(".")

from canoe import def_species, load_configure, index_map
from canoe.snap import def_thermo
from canoe.athena import Mesh, ParameterInput, Outputs, MeshBlock

def set_atmos_run_RT(qNH3: float, 
                     T0: float = 180.0, 
                     RHmax: float=1.0,
                     adlnNH3dlnP: float=0.0,
                     pmin: float = 0.0, 
                     pmax: float = 0.0
                     ):  
    ## construct atmos with a rh limit
    mb.construct_atmosphere(pin, qNH3, T0, RHmax)

    ## modify the top humidity with a increment
    mb.modify_dlnNH3dlnP_rhmax(adlnNH3dlnP, pmin, pmax, RHmax)

    ## do radiative transfer
    rad = mb.get_rad()
    rad.cal_radiance(mb, mb.k_st, mb.j_st)

    nb = rad.get_num_bands()
    tb = np.array([0.0] * 4 * nb)

    for ib in range(nb):
        toa = rad.get_band(ib).get_toa()[0]
        tb[ib * 4 : ib * 4 + 4] = toa
    return tb[4:]


## main
if __name__ == "__main__":
    h5=h5py.File('run_mcmc_adlnNH3_background_2000.h5', 'r')
    chain = np.array(h5['mcmc']['chain'][:])  ## [nstep,nwlk,ndim]
    h5.close()
 
    [nstep,nwlk,ndim]=chain.shape
    nwlk=1
    tb=np.zeros((nstep,nwlk,20))
    SH_NH3=np.zeros((nstep,nwlk,1600))
    RH_NH3=np.zeros((nstep,nwlk,1600))

    global pin
    pin = ParameterInput()
    pin.load_from_file("juno_mwr.inp")

    vapors = pin.get_string("species", "vapor").split(", ")
    clouds = pin.get_string("species", "cloud").split(", ")
    tracers = pin.get_string("species", "tracer").split(", ")

    def_species(vapors=vapors, clouds=clouds, tracers=tracers)
    def_thermo(pin)

    config = load_configure("juno_mwr.yaml")

    mesh = Mesh(pin)
    mesh.initialize(pin)

    global mb
    mb = mesh.meshblock(0)

    # aircolumn=mb.get_aircolumn(mb.k_st,mb.j_st,mb.i_st,mb.i_ed)    
    # print(len(aircolumn))

    pindex = index_map.get_instance()
    iNH3 = pindex.get_vapor_id("NH3")
    # iH2O = pindex.get_vapor_id("H2O")

    # for i in range(len(aircolumn)):
    #     ap=aircolumn[i]

    #     ## mass fraction kg/kg
    #     sh_NH3=ap.hydro()[iNH3]

        ##density
        # rho=ap.hydro()[0]

        # ap_mole=ap.to_mole_fraction()
        # rh=ap_mole.get_rh(iNH3)

    # print(pin.get_string("job", "problem_id"))
    # out = Outputs(mesh, pin)
    # out.make_outputs(mesh, pin)

    for istep in range(nstep):
        print(istep)
        for iwalk in range(nwlk):
            [nh3, temperature,RHmax, adlnNH3dlnP,Pmax]=chain[istep, iwalk, :]
            # print(nh3,temperature,RHmax)  
            tb[istep,iwalk,:] = set_atmos_run_RT(nh3, temperature, RHmax, adlnNH3dlnP, 1.E-3, Pmax)
            # print( tb[istep,iwalk,:])
            aircolumn=mb.get_aircolumn(mb.k_st,mb.j_st,mb.i_st,mb.i_ed)   
            for i in range(len(aircolumn)):
                ap=aircolumn[i]
                SH_NH3[istep,iwalk,i]=ap.hydro()[iNH3]

                ap_mole=ap.to_mole_fraction()
                RH_NH3[istep,iwalk,i]=ap_mole.get_rh(iNH3)

        # exit()

    H5OUT=h5py.File('TB_adlnNH3_background_2000.h5', 'w')
    H5OUT.create_dataset('tb', data=tb)
    H5OUT.close()

 
    profOUT=h5py.File('NH3_adlnNH3_background_2000.h5', 'w')
    profOUT.create_dataset('SH_NH3', data=SH_NH3)
    profOUT.create_dataset('RH_NH3', data=RH_NH3)
    profOUT.close()   