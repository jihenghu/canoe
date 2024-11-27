#ifndef SRC_UTILS_CONSTRUCT_ATMOSPHERE_DRY_HPP_
#define SRC_UTILS_CONSTRUCT_ATMOSPHERE_DRY_HPP_

// C/C++
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

// set up an adiabatic atmosphere
void construct_atmosphere(MeshBlock *pmb, ParameterInput *pin, Real xNH3,
                          Real T0, Real rh_max_nh3, int Jindex, std::string method="dry", Real H2Oppmv=2500, int max_iter=200);
// set up an adiabatic atmosphere with given Ts
void construct_atmosphere_Ts(MeshBlock *pmb, ParameterInput *pin, Real NH3ppmv,
                          Real Ts, Real rh_max_nh3, int Jindex, std::string method="dry", Real H2Oppmv=2500);
// return reference temperature for given Ts and qXX
Real derive_T1bar_given_Ts(MeshBlock *pmb, ParameterInput *pin, Real NH3ppmv, 
                          Real Ts, std::string method="dry", Real H2Oppmv=2500);
//return bottom temperature Ts given Theta, T0 for dry adiabatic
Real retrieve_Ts_given_T1bar(MeshBlock *pmb, ParameterInput *pin, Real NH3ppmv,
                          Real T0, Real rh_max_nh3, int Jindex, std::string method="dry", Real H2Oppmv=2500, int max_iter=200); 
#endif  // SRC_UTILS_CONSTRUCT_ATMOSPHERE_HPP_
