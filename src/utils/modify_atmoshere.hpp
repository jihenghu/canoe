#ifndef SRC_UTILS_MODIFY_ATMOSPHERE_HPP_
#define SRC_UTILS_MODIFY_ATMOSPHERE_HPP_

// C/C++
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

// helper functions, will be moved in the future
int find_pressure_level_lesser_pybind(Real pres, AthenaArray<Real> const &w,
                                      int k, int j, int is, int ie);

// modify atmoshere with adlnNH3dlnP with a RH_max limit
void modify_atmos_adlnNH3dlnP_RHmax(MeshBlock *pmb, ParameterInput *pin, Real adlnNH3dlnP, 
                              Real pmin, Real pmax, Real rhmax, int Jindex, std::string method="dry");

// modify atmoshere with adlnTdlnP
void modify_atmos_adlnTdlnP(MeshBlock *pmb, ParameterInput *pin, Real adlnTdlnP, 
                              Real pmin, Real pmax, int Jindex, std::string method="dry");

// overwrite e- , Na
void set_tracer_layer(MeshBlock *pmb, int tracer_id, int j, int i,  double value);

#endif  // SRC_UTILS_MODIFY_ATMOSPHERE_HPP_
