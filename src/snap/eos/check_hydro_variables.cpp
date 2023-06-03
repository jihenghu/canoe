// canoe
#include <configure.hpp>

// athena
#include <athena/athena.hpp>
#include <athena/coordinates/coordinates.hpp>
#include <athena/hydro/hydro.hpp>
#include <athena/hydro/srcterms/hydro_srcterms.hpp>
#include <athena/mesh/mesh.hpp>
#include <athena/stride_iterator.hpp>

// debugger
#include <debugger/debugger.hpp>

// snap
#include "../meshblock_impl.hpp"
#include "../thermodynamics/thermodynamics.hpp"

void check_hydro_variables(MeshBlock *pmb, AthenaArray<Real> const &w) {
  for (int k = pmb->ks; k <= pmb->ke; ++k)
    for (int j = pmb->js; j <= pmb->je; ++j)
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        for (int n = 0; n <= NHYDRO; ++n) {
          if (w(n, k, j, i) < 0.) {
            Debugger::Fatal("check_hydro_variables", "density", "negative");
          }
        }

        if (NON_BAROTROPIC_EOS) {
          if (w(IPR, k, j, i) < 0.) {
            Debugger::Fatal("check_hydro_variables", "pressure", "negative");
          }
        }

        Thermodynamics *pthermo = pmb->pimpl->pthermo;
        Real temp = pthermo->GetTemp(w.at(k, j, i));
        Real grav = -pmb->phydro->hsrc.GetG1();
        if (grav != 0) {
          Real Tmin = grav * pmb->pcoord->dx1f(i) / pthermo->GetRd();
          if (temp < Tmin) {
            Debugger::Fatal("check_hydro_variables", "temperature", "low");
          }
        }
      }

  // make a copy of w, needed for outflow boundary condition
  // w1 = w;
  Debugger::Print("Hydro check passed.");
}
