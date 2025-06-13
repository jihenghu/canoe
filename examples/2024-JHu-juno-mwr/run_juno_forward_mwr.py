#! /usr/bin/env python3
import numpy as np
import sys
import h5py

sys.path.append("../python")  # Adjust the path to build/python/
sys.path.append(".")

## load the necessary modules
from canoe import def_species, load_configure, index_map
from canoe.snap import def_thermo
from canoe.athena import Mesh, ParameterInput, Outputs, MeshBlock
from canoe.harp import radiation_band, radiation


## ========================================================================
##                       configure the Juno MWR forward model
##--------------------------------------------------------------------------
# load the configuration file
pin = ParameterInput()
pin.load_from_file("juno_mwr.inp")
pin.set_boolean("job", "verbose", False)  

# register the speicies and thermodynamics
vapors = pin.get_string("species", "vapor").split(", ")
clouds = pin.get_string("species", "cloud").split(", ")
tracers = pin.get_string("species", "tracer").split(", ")

def_species(vapors=vapors, clouds=clouds, tracers=tracers)
def_thermo(pin)

# load bands and opacities
config = load_configure("juno_mwr.yaml")


# set incident angles 
angle =[0.0, 15, 30.0, 45.]  # in degrees
angles = ' '.join([f"({x},)" for x in angle])
pin.set_string("radiation","outdir",angles)
print(pin.get_string("radiation","outdir"))


# set the gravity 
pin.set_string("hydro", "grav_acc1", f"{-23.3}") # Equatorial zone
# pin.set_string("hydro", "grav_acc1", f"{-27.01}") # pole 
print("Gravity acceleration:", pin.get_string("hydro", "grav_acc1"))


# get index_map 
pindex = index_map.get_instance()
iNH3 = pindex.get_vapor_id("NH3")
iH2O = pindex.get_vapor_id("H2O")
print(f"iNH3 = {iNH3}, iH2O = {iH2O}")

P0 = pin.get_real("mesh", "ReferencePressure")
print(f"Reference Pressure: {P0} Pa")


# set mesh; we use the 1st column of the 1st meshblock of the Athena++ mesh
nx2 = 1  # air column number, single column in this case
pin.set_string("mesh", "nx2", f"{nx2}")

# init the Mesh
mesh = Mesh(pin) 
mesh.initialize(pin)

# get the first meshblock
mb = mesh.meshblock(0)  


##=============================================================================================
##                  construct atmosphere: moist adiabate + constant NH3
##---------------------------------------------------------------------------------------------

# EZ abundances and temperature in Cheng (2020) 
xNH3=351     # deep-layer abundance, ppmv
xH2O=2500    # ppmv
T1bar=169    # 1-bar temperature, K

## use the first ap column of the block
Jindex=0   # Jindex ∈ [0, nx2-1]

## RH limit 
RHmax=1  # limit of relative humidity, 1 means 100%, only affects NH3 cloud-layer, set to 1 in EZ
maxint=200  # do not change

## a moist adiabatic uniform NH3 profile, with RHlimit
adiabate="pseudo"  # or "dry"

# construct the atmosphere
mb.construct_atmosphere(pin, xNH3, T1bar, RHmax, Jindex, adiabate, xH2O, maxint)

# apply a NH3 gradient
# adlnNH3dlnP=-0.08
# pmax=4.78E5  # pa
# pmin=1E-3    # pa
# mb.modify_dlnNH3dlnP_rhmax(pin, adlnNH3dlnP, pmin, pmax, RHmax, Jindex, adiabate) 


## ===============================================================================================
##                              calculate the profile  
## -----------------------------------------------------------------------------------------------

## calc for the Jindex_st column profile
aircolumn = mb.get_aircolumn(mb.k_st, mb.j_st + Jindex, mb.i_st, mb.i_ed)

#### loop over layers bottom up
nlyr=len(aircolumn)             ## default 1600 pressure layers

SH_NH3= np.array([0.0]*nlyr)    ## specific mixing ratio, kg/kg
RH_NH3= np.array([0.0]*nlyr)    ## relative humidity
# SH_H2O= np.array([0.0]*nlyr)    ## specific mixing ratio, kg/kg
# RH_H2O= np.array([0.0]*nlyr)    ## relative humidity

NH3_ppmv= np.array([0.0]*nlyr)   ## mole fraction, eg., 380 ppmv
# H2O_ppmv= np.array([0.0]*nlyr)   ## mole fraction, eg., 2500 ppmv

temp= np.array([0.0]*nlyr)      ## temperature, K
theta= np.array([0.0]*nlyr)     ## potential temperature, K, reference pressure P0 = 1 bar

for i in range(nlyr): 
    ap = aircolumn[i]
    SH_NH3[i] = ap.hydro()[iNH3]
    # SH_H2O[i] = ap.hydro()[iH2O]
    
    ap_mole = ap.to_mole_fraction()
    RH_NH3[i] = ap_mole.get_rh(iNH3)
    # RH_H2O[i] = ap_mole.get_rh(iH2O)

    NH3_ppmv[i] = ap_mole.hydro()[iNH3]*1E6  
    # H2O_ppmv[i] = ap_mole.hydro()[iH2O]*1E6  

    ## temp and theta should be calculated before applying NH3 gradient, or it will be wrong.
    temp[i] = mb.get_temp(mb.k_st, mb.j_st + Jindex,i)    
    theta[i] = mb.get_theta(P0, mb.k_st, mb.j_st + Jindex,i)

# Save the results to an HDF5 file
with h5py.File('juno_mwr_fwd_case-EZ-moist-profile.h5', 'w') as h5file:
    h5file.create_dataset('SH_NH3', data=SH_NH3)
    h5file.create_dataset('RH_NH3', data=RH_NH3)
    h5file.create_dataset('NH3_ppmv', data=NH3_ppmv)
    h5file.create_dataset('temp', data=temp)
    h5file.create_dataset('theta', data=theta)

## ===============================================================================================
##                           calculate the radiance
## -----------------------------------------------------------------------------------------------

# get radance obj
rad = mb.get_rad()
nband = rad.get_num_bands()

# calculate the radiance for the Jindex_st column
rad.cal_radiance(mb, mb.k_st, mb.j_st + Jindex)
nray=len(angle)
tb = np.array([0.0] * nray * nband)

print(f"Number of bands: {nband}, Number of angles: {nray}")

for ib in range(nband):
    toa = rad.get_band(ib).get_toa()[0]
    tb[ib * nray : (ib+1) * nray] = toa

with h5py.File('juno_mwr_fwd_case-EZ-moist-radiance.h5', 'w') as h5file:
    h5file.create_dataset('tb', data=tb)
    h5file.create_dataset('angle', data=angle)


## -------------   use-inbuilt default output for more variables   ------------------

print(pin.set_string("job", "problem_id","juno_mwr_fwd_case-EZ-moist"))
out = Outputs(mesh, pin)
out.make_outputs(mesh, pin)
import os
os.system("./combine.py")

## ---------------------------------------------------------------