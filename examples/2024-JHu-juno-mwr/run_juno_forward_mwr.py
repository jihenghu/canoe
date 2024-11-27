#! /usr/bin/env python3
import numpy as np
import sys
import h5py

sys.path.append("../python")
sys.path.append(".")

from canoe import def_species, load_configure, index_map
from canoe.snap import def_thermo
from canoe.athena import Mesh, ParameterInput, Outputs, MeshBlock
# from canoe.harp import radiation_band, radiation

pin = ParameterInput()
pin.load_from_file("juno_mwr.inp")

vapors = pin.get_string("species", "vapor").split(", ")
clouds = pin.get_string("species", "cloud").split(", ")
tracers = pin.get_string("species", "tracer").split(", ")

def_species(vapors=vapors, clouds=clouds, tracers=tracers)
def_thermo(pin)

config = load_configure("juno_mwr.yaml")

nx2 = 1 
pin.set_boolean("job", "verbose", False)
pin.set_string("mesh", "nx2", f"{nx2}")

pindex = index_map.get_instance()
iNH3 = pindex.get_vapor_id("NH3")

mesh = Mesh(pin) 
mesh.initialize(pin)

global mb, rad, nb
mb = mesh.meshblock(0)  ## fetch the first block of the mesh
P0 = pin.get_real("mesh", "ReferencePressure")

## construct atmosphere
## deep layer params
qNH3=381.6
qH2O=2500
T1bar=177.6

## use the first ap column of the block
Jindex=0 

## RH limit 
RHmax=0.63

## a dry adiabatic uniform NH3 profile, with RHlimit
mb.construct_atmosphere(pin, qNH3, T1bar, RHmax, 0,"dry",qH2O, 200)

## apply a Temp gradient, shoud go before implementing NH3 gradient
adlnT=-0.04
pmax=2E5
pmin=0.5E5
mb.modify_dlnTdlnP(pin, adlnT, pmin, pmax, Jindex, "dry") 

# apply a NH3 gradient
adlnNH3dlnP=-0.08
pmax=4.78E5
pmin=1E-3
mb.modify_dlnNH3dlnP_rhmax(pin, adlnNH3dlnP, pmin, pmax, RHmax, Jindex, "dry") 


# ### calc for the Jindex_st column profile
aircolumn = mb.get_aircolumn(mb.k_st, mb.j_st + Jindex, mb.i_st, mb.i_ed)   
#### loop over layers bottom up
nlyr=len(aircolumn)
SH_NH3= np.array([0.0]*nlyr) ## specific mixing ratio, kg/kg
RH_NH3= np.array([0.0]*nlyr) ## relative humidity
molefrc= np.array([0.0]*nlyr) ## mole fraction, eg., 0.000380
temp= np.array([0.0]*nlyr) ## mole fraction, eg., 0.000380
theta= np.array([0.0]*nlyr) ## mole fraction, eg., 0.000380

for i in range(nlyr): 
    ap = aircolumn[i]
    SH_NH3[i] = ap.hydro()[iNH3]
    ap_mole = ap.to_mole_fraction()
    RH_NH3[i] = ap_mole.get_rh(iNH3)
    molefrc[i] = ap_mole.hydro()[iNH3]*1E6  ## ppm

    ## temp and theta should be calculated before applying NH3 gradient, or it will be wrong.
    temp[i] = mb.get_temp(mb.k_st, mb.j_st + Jindex,i)    
    theta[i] = mb.get_theta(P0, mb.k_st, mb.j_st + Jindex,i)

# # Save the results to an HDF5 file
# with h5py.File('juno_mwr_NH3_profile.h5', 'w') as h5file:
#     h5file.create_dataset('SH_NH3', data=SH_NH3)
#     h5file.create_dataset('RH_NH3', data=RH_NH3)
#     h5file.create_dataset('molefrc', data=molefrc)
#     h5file.create_dataset('temp', data=temp)
#     h5file.create_dataset('theta', data=theta)

## get all radiance，4 outdirections
# rad = mb.get_rad()
# nb = rad.get_num_bands()
# rad.cal_radiance(mb, mb.k_st, mb.j_st + Jindex)
# tb = np.array([0.0] * 4 * nb)

# for ib in range(nb):
#     toa = rad.get_band(ib).get_toa()[0]
#     tb[ib * 4 : ib * 4 + 4] = toa

## calc radiance
rad = mb.get_rad()
rad.cal_radiance(mb, mb.k_st, mb.j_st + Jindex)

## output into nc
print(pin.set_string("job", "problem_id","juno_mwr_dry_gradient_case"))
out = Outputs(mesh, pin)
out.make_outputs(mesh, pin)
