// pybind
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// athena
#include <athena/athena.hpp>
#include <athena/parameter_input.hpp>

// canoe
#include <configure.hpp>
#include <constants.hpp>
#include <index_map.hpp>
#include <variable.hpp>

// harp
#include <harp/absorber.hpp>
#include <harp/radiation.hpp>
#include <harp/radiation_band.hpp>

namespace py = pybind11;

PYBIND11_MODULE(pycanoe, m) {
  m.attr("__name__") = "pycanoe";
  m.doc() = "Python bindings for CANOE";

  // Constants
  py::module m_constants = m.def_submodule("constants");
  m_constants.attr("Rgas") = Constants::Rgas;
  m_constants.attr("Rgas_cgs") = Constants::Rgas_cgs;
  m_constants.attr("kBoltz") = Constants::kBoltz;
  m_constants.attr("kBoltz_cgs") = Constants::kBoltz_cgs;
  m_constants.attr("Lo") = Constants::Lo;
  m_constants.attr("hPlanck") = Constants::hPlanck;
  m_constants.attr("hPlanck_cgs") = Constants::hPlanck_cgs;
  m_constants.attr("cLight") = Constants::cLight;
  m_constants.attr("cLight_cgs") = Constants::cLight_cgs;
  m_constants.attr("stefanBoltzmann") = Constants::stefanBoltzmann;

  // IndexMap
  py::class_<IndexMap>(m, "index_map")
      .def_static("get_instance", &IndexMap::GetInstance)
      .def_static("init_from_athena_input", &IndexMap::InitFromAthenaInput)

      .def("get_vapor_id", &IndexMap::GetVaporId)
      .def("get_cloud_id", &IndexMap::GetCloudId)
      .def("get_tracer_id", &IndexMap::GetTracerId)
      .def("get_species_id", &IndexMap::GetSpeciesId);

  // Variable type
  py::enum_<Variable::Type>(m, "VariableType")
      .value("MassFrac", Variable::Type::MassFrac)
      .value("MassConc", Variable::Type::MassConc)
      .value("MoleFrac", Variable::Type::MoleFrac)
      .value("MoleConc", Variable::Type::MoleConc)
      .export_values();

  // Variable
  py::class_<Variable>(m, "AirParcel")
      .def(py::init<>())

      .def("hydro",
           [](const Variable& var) {
             py::array_t<double> result(NCLOUD, var.c);
             return result;
           })

      .def("cloud",
           [](const Variable& var) {
             py::array_t<double> result(NCLOUD, var.c);
             return result;
           })

      .def("tracer",
           [](const Variable& var) {
             py::array_t<double> result(NCLOUD, var.x);
             return result;
           })

      .def("to_mass_fraction", &Variable::ToMassFraction)
      .def("to_mass_concentration", &Variable::ToMassConcentration)
      .def("to_mole_fraction", &Variable::ToMoleFraction)
      .def("to_mole_concentration", &Variable::ToMoleConcentration);
}
