// C/C++
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

// application
#include <application/application.hpp>
// athena
#include <athena/coordinates/coordinates.hpp>
#include <athena/eos/eos.hpp>
#include <athena/field/field.hpp>
#include <athena/hydro/hydro.hpp>
#include <athena/mesh/mesh.hpp>
#include <athena/parameter_input.hpp>
#include <athena/scalars/scalars.hpp>
// canoe
#include <air_parcel.hpp>
#include <constants.hpp>
#include <impl.hpp>
#include <index_map.hpp>
#include <tracer/tracer.hpp>


// snap
#include <snap/thermodynamics/thermodynamics.hpp>
#include <snap/thermodynamics/atm_thermodynamics.hpp>

// helper functions, will be moved in the future
int find_pressure_level_lesser_pybind(Real pres, AthenaArray<Real> const &w,
                                      int k, int j, int is, int ie) {
  for (int i = is; i <= ie; ++i)
    if (w(IPR, k, j, i) < pres) return i;

  return ie + 1;
};

// modify atmoshere with adlnNH3dlnP with a RH_max limit
void modify_atmos_adlnNH3dlnP_RHmax(MeshBlock *pmb, ParameterInput *pin, Real adlnNH3dlnP, 
                              Real pmin, Real pmax, Real rhmax, int Jindex, std::string method="dry") {
  int is = pmb->is, js = pmb->js, ks = pmb->ks;
  int ie = pmb->ie, je = pmb->je, ke = pmb->ke;
  ke = ks;
  js = js+Jindex;
  je = js;

  Hydro *phydro = pmb->phydro;
  auto pthermo = Thermodynamics::GetInstance();
  auto pcoord = pmb->pcoord;
  Real H0 = pcoord->GetPressureScaleHeight();
  Real dlnp = pcoord->dx1f(is) / H0;
  Real Tmin = pin->GetReal("problem", "Tmin");

  auto pindex = IndexMap::GetInstance();
  int iNH3 = pindex->GetVaporId("NH3");

  // loop over all aircolumns
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j) {
      int ibegin =
          find_pressure_level_lesser_pybind(pmax, phydro->w, k, j, is, ie);
      int iend =
          find_pressure_level_lesser_pybind(pmin, phydro->w, k, j, is, ie);

      auto &&air = AirParcelHelper::gather_from_primitive(pmb, k, j, ibegin);
      air.ToMoleFraction();

      for (int i = ibegin; i < iend; ++i) {
        auto &&air1 = AirParcelHelper::gather_from_primitive(pmb, k, j, i+1);
        air1.ToMoleFraction();

        pthermo->Extrapolate(&air, -dlnp, method);

        air.w[IDN]= air1.w[IDN]; // keep temperature as it is.
        air.w[iNH3] += adlnNH3dlnP * air.w[iNH3] * dlnp;
        auto rates = pthermo->TryEquilibriumTP_VaporCloud(air, iNH3);
        air.w[iNH3] += rates[0];

        // adjust qNH3 according to RH_max
        Real rh = get_relative_humidity(air, iNH3);
        air.w[iNH3] *= std::min(rhmax / rh, 1.);
        
        AirParcelHelper::distribute_to_primitive(pmb, k, j, i + 1, air);
      }
    }
};

// modify atmoshere with adlnTdlnP with a RH_max limit
void modify_atmos_adlnTdlnP(MeshBlock *pmb, ParameterInput *pin, Real adlnTdlnP, 
                              Real pmin, Real pmax, int Jindex, std::string method="dry") {
  int is = pmb->is, js = pmb->js, ks = pmb->ks;
  int ie = pmb->ie, je = pmb->je, ke = pmb->ke;
  ke = ks;
  js = js+Jindex;
  je = js;

  Hydro *phydro = pmb->phydro;
  auto pthermo = Thermodynamics::GetInstance();
  auto pcoord = pmb->pcoord;
  Real H0 = pcoord->GetPressureScaleHeight();
  Real dlnp = pcoord->dx1f(is) / H0;

  auto pindex = IndexMap::GetInstance();
  int iNH3 = pindex->GetVaporId("NH3");

  Real Tmin = pin->GetReal("problem", "Tmin");

  // loop over all aircolumns
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j) {
      int ibegin =
          find_pressure_level_lesser_pybind(pmax, phydro->w, k, j, is, ie);
      int iend =
          find_pressure_level_lesser_pybind(pmin, phydro->w, k, j, is, ie);

      auto &&air = AirParcelHelper::gather_from_primitive(pmb, k, j, ibegin);
      air.ToMoleFraction();

      for (int i = ibegin; i < iend; ++i) {
        pthermo->Extrapolate(&air, -dlnp, method, 0, adlnTdlnP);
        // air.w[IDN] += adlnTdlnP * air.w[IDN] * dlnp; //deplicated
        air.w[IDN]= std::max(air.w[IDN], Tmin);    
        AirParcelHelper::distribute_to_primitive(pmb, k, j, i + 1, air);
      }

      for (int i = iend; i <= ie; ++i) {
        pthermo->Extrapolate(&air, -dlnp, method);
        air.w[IDN]= std::max(air.w[IDN], Tmin);  
        AirParcelHelper::distribute_to_primitive(pmb, k, j, i + 1, air);
      }
    }
};

// overwrite e- , Na
void set_tracer_layer(MeshBlock *pmb, int tracer_id, int j, int i, double value) {
  int is = pmb->is, js = pmb->js, ks = pmb->ks;
  int ie = pmb->ie, je = pmb->je, ke = pmb->ke;
  // ke = ks;
  // js = js+Jindex;
  // je = js;

  auto pimpl = pmb->pimpl;
  auto phydro = pmb->phydro;
  auto ptracer = pimpl->ptracer;
  auto pthermo = Thermodynamics::GetInstance();
  auto pcoord = pmb->pcoord;

  for (int k = ks; k <= ke; ++k) {
        ptracer->u(tracer_id, k, j, i) = value;
      }

  auto peos = pmb->peos;
  auto pfield = pmb->pfield;
  auto pscalars = pmb->pscalars;
  auto pbval = pmb->pbval;

  // primitive to conserved conversion (hydro)
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie,
                             js, je, ks, ke);

  // conserved to primitive conversion (tracer)
  peos->PassiveScalarConservedToPrimitive(pscalars->s, phydro->u, pscalars->r,
                                          pscalars->r, pcoord, is, ie, js, je,
                                          ks, ke);
};