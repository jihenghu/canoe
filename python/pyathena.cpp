// pybind
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// athena
#include <athena/athena.hpp>
#include <athena/mesh/mesh.hpp>
#include <athena/outputs/outputs.hpp>
#include <athena/parameter_input.hpp>

// canoe
#include <impl.hpp>

// harp
#include <harp/radiation.hpp>

// utils
#include <utils/construct_atmosphere.hpp>
#include <utils/modify_atmoshere.hpp>
// tracer
#include <tracer/tracer.hpp>
// snap
#include <snap/thermodynamics/thermodynamics.hpp>

namespace py = pybind11;

void init_athena(py::module &parent) {
  auto m = parent.def_submodule("athena", "Python bindings for Athena++");

  m.def("nghost", []() { return NGHOST; });

  py::enum_<IOWrapper::FileMode>(m, "FileMode")
      .value("read", IOWrapper::FileMode::read)
      .value("write", IOWrapper::FileMode::write)
      .export_values();

  py::class_<ParameterInput>(m, "ParameterInput")
      .def(py::init())
      .def("load_from_file",
           [](ParameterInput &pin, const std::string &filename) {
             IOWrapper infile;
             infile.Open(filename.c_str(), IOWrapper::FileMode::read);
             pin.LoadFromFile(infile);
             infile.Close();
           })
      .def("get_integer", &ParameterInput::GetInteger)
      .def("get_real", &ParameterInput::GetReal)
      .def("get_boolean", &ParameterInput::GetBoolean)
      .def("set_boolean", &ParameterInput::SetBoolean)
      .def("get_string", &ParameterInput::GetString)
      .def("set_string", &ParameterInput::SetString)
      .def("does_parameter_exist", &ParameterInput::DoesParameterExist);

  // AthenArray
  py::class_<AthenaArray<Real>>(m, "AthenaArray", py::buffer_protocol())
      .def_buffer([](AthenaArray<Real> &m) -> py::buffer_info {
        size_t stride4 = m.GetDim1() * m.GetDim2() * m.GetDim3() * sizeof(Real);
        size_t stride3 = m.GetDim1() * m.GetDim2() * sizeof(Real);
        size_t stride2 = m.GetDim1() * sizeof(Real);
        size_t stride1 = sizeof(Real);
        if (m.GetDim4() > 1) {
          return py::buffer_info(
              // Pointer to buffer
              m.data(),
              // Size of one scalar
              sizeof(Real),
              // Python struct-style format descriptor
              py::format_descriptor<Real>::format(),
              // Number of dimensions
              4,
              // Buffer dimensions
              {m.GetDim4(), m.GetDim3(), m.GetDim2(), m.GetDim1()},
              // Strides (in bytes) for each index
              {stride4, stride3, stride2, stride1});
        } else if (m.GetDim3() > 1) {
          return py::buffer_info(m.data(), sizeof(Real),
                                 py::format_descriptor<Real>::format(), 3,
                                 {m.GetDim3(), m.GetDim2(), m.GetDim1()},
                                 {stride3, stride2, stride1});
        } else if (m.GetDim2() > 1) {
          return py::buffer_info(
              m.data(), sizeof(Real), py::format_descriptor<Real>::format(), 2,
              {m.GetDim2(), m.GetDim1()}, {stride2, stride1});
        } else {
          return py::buffer_info(m.data(), sizeof(Real),
                                 py::format_descriptor<Real>::format(), 1,
                                 {m.GetDim1()}, {stride1});
        }
      });

  // RegionSize
  py::class_<RegionSize>(m, "RegionSize")
      .def_property(
          "x1min", [](RegionSize const &rs) { return rs.x1min; },
          [](RegionSize &rs, Real x1min) { rs.x1min = x1min; })

      .def_property(
          "x2min", [](RegionSize const &rs) { return rs.x2min; },
          [](RegionSize &rs, Real x2min) { rs.x2min = x2min; })

      .def_property(
          "x3min", [](RegionSize const &rs) { return rs.x3min; },
          [](RegionSize &rs, Real x3min) { rs.x3min = x3min; })

      .def_property(
          "x1max", [](RegionSize const &rs) { return rs.x1max; },
          [](RegionSize &rs, Real x1max) { rs.x1max = x1max; })

      .def_property(
          "x2max", [](RegionSize const &rs) { return rs.x2max; },
          [](RegionSize &rs, Real x2max) { rs.x2max = x2max; })

      .def_property(
          "x3max", [](RegionSize const &rs) { return rs.x3max; },
          [](RegionSize &rs, Real x3max) { rs.x3max = x3max; })

      .def_property(
          "nx1", [](RegionSize const &rs) { return rs.nx1; },
          [](RegionSize &rs, int nx1) { rs.nx1 = nx1; })

      .def_property(
          "nx2", [](RegionSize const &rs) { return rs.nx2; },
          [](RegionSize &rs, int nx2) { rs.nx2 = nx2; })

      .def_property(
          "nx3", [](RegionSize const &rs) { return rs.nx3; },
          [](RegionSize &rs, int nx3) { rs.nx3 = nx3; })

      .def_property(
          "x1rat", [](RegionSize const &rs) { return rs.x1rat; },
          [](RegionSize &rs, Real x1rat) { rs.x1rat = x1rat; })

      .def_property(
          "x2rat", [](RegionSize const &rs) { return rs.x2rat; },
          [](RegionSize &rs, Real x2rat) { rs.x2rat = x2rat; })

      .def_property(
          "x3rat", [](RegionSize const &rs) { return rs.x3rat; },
          [](RegionSize &rs, Real x3rat) { rs.x3rat = x3rat; });

  // Mesh
  py::class_<Mesh>(m, "Mesh")
      .def(py::init<ParameterInput *, int>(), py::arg("pin"),
           py::arg("mesh_only") = false)

      .def("initialize",
           [](Mesh &mesh, ParameterInput *pin) {
             bool restart = false;

             // set up components
             for (int b = 0; b < mesh.nblocal; ++b) {
               MeshBlock *pmb = mesh.my_blocks(b);
               pmb->pimpl = std::make_shared<MeshBlock::Impl>(pmb, pin);
             }
             mesh.Initialize(restart, pin);
           })

      .def("meshblocks",
           [](Mesh &mesh) {
             py::list lst;
             for (size_t i = 0; i < mesh.nbtotal; ++i) {
               lst.append(mesh.my_blocks(i));
             }
             return lst;
           })

      .def(
          "meshblock",
          [](Mesh &mesh, int n) {
            if (n < 0 || n >= mesh.nbtotal) {
              throw py::index_error();
            }
            return mesh.my_blocks(n);
          },
          py::return_value_policy::reference);

  // MeshBlock
  py::class_<MeshBlock>(m, "MeshBlock")
      .def_readonly("block_size", &MeshBlock::block_size)

      .def_readonly("i_st", &MeshBlock::is)
      .def_readonly("i_ed", &MeshBlock::ie)
      .def_readonly("j_st", &MeshBlock::js)
      .def_readonly("j_ed", &MeshBlock::je)
      .def_readonly("k_st", &MeshBlock::ks)
      .def_readonly("k_ed", &MeshBlock::ke)

      //.def_readonly("inversion", [](MeshBlock const& pmb) {
      //  return pmb.pimpl->all_fits;
      //});

      .def("modify_dlnNH3dlnP_rhmax",
           [](MeshBlock &mesh_block, ParameterInput *pin, Real adlnNH3dlnP, Real pmin, Real pmax, 
              Real rhmax, int Jindex, std::string method="dry") {
             return modify_atmos_adlnNH3dlnP_RHmax(&mesh_block, pin, adlnNH3dlnP, pmin,
                                                 pmax, rhmax, Jindex, method);
           },
            py::arg("pin"), py::arg("adlnNH3dlnP"), py::arg("pmin"), py::arg("pmax"), 
            py::arg("rhmax"), py::arg("Jindex"), 
            py::arg("method") = "dry",
            "Modify atmosphere with adlnNH3dlnP and RH_NH3 limit.")

      .def("modify_dlnTdlnP",
           [](MeshBlock &mesh_block,  ParameterInput *pin, Real adlnTdlnP, Real pmin, Real pmax, 
              int Jindex, std::string method="dry") {
             return modify_atmos_adlnTdlnP(&mesh_block, pin, adlnTdlnP, pmin,
                                                 pmax, Jindex, method);
           },
            py::arg("pin"), py::arg("adlnTdlnP"), py::arg("pmin"), py::arg("pmax"), 
            py::arg("Jindex"), 
            py::arg("method") = "dry",
            "Modify atmosphere with Temperature gradient adlnTdlnP.")

      .def("construct_atmosphere", 
            [](MeshBlock &mesh_block, ParameterInput *pin, Real xNH3, Real T0, 
              Real rh_max_nh3, int Jindex, std::string method = "dry", Real H2Oppmv=2500, int max_iter=200) {
                // Call the actual C++ function
              return construct_atmosphere(&mesh_block, pin, xNH3, T0, rh_max_nh3, Jindex, method, H2Oppmv, max_iter);
            },
            py::arg("pin"), py::arg("xNH3"), py::arg("T0"), 
            py::arg("rh_max_nh3"), py::arg("Jindex"), 
            py::arg("method") = "dry",
            py::arg("H2Oppmv") = 2500,
            py::arg("max_iter") = 200,
            "Construct the atmosphere for the given MeshBlock with specified parameters.")

      .def("construct_atmosphere_Ts", 
            [](MeshBlock &mesh_block, ParameterInput *pin, Real xNH3, Real Ts, 
              Real rh_max_nh3, int Jindex, std::string method = "dry", Real H2Oppmv=2500) {
                // Call the actual C++ function
              return construct_atmosphere_Ts(&mesh_block, pin, xNH3, Ts, rh_max_nh3, Jindex, method, H2Oppmv);
            },
            py::arg("pin"), py::arg("xNH3"), py::arg("Ts"), 
            py::arg("rh_max_nh3"), py::arg("Jindex"), 
            py::arg("method") = "dry",
            py::arg("H2Oppmv") = 2500,
            "Construct the atmosphere for the given MeshBlock with specified parameters.")
      
      .def("derive_T1bar_given_Ts", 
            [](MeshBlock &mesh_block, ParameterInput *pin, Real xNH3, Real Ts, 
              std::string method = "dry", Real H2Oppmv=2500) {
                // Call the actual C++ function
              return derive_T1bar_given_Ts(&mesh_block, pin, xNH3, Ts, method, H2Oppmv);
            },
            py::arg("pin"),  py::arg("xNH3"), py::arg("Ts"), 
            py::arg("method") = "dry", 
            py::arg("H2Oppmv")=2500,
            "Return Reference Temperature [1 Bar] for the given MeshBlock with specified parameters.")

      .def("retrieve_Ts_given_T1bar", 
            [](MeshBlock &mesh_block, ParameterInput *pin, Real xNH3, Real T0, 
              Real rh_max_nh3, int Jindex, std::string method = "dry", Real H2Oppmv=2500, int max_iter=200) {
                // Call the actual C++ function
              return retrieve_Ts_given_T1bar(&mesh_block, pin, xNH3, T0, rh_max_nh3, Jindex, method, H2Oppmv, max_iter);
            },
            py::arg("pin"), py::arg("xNH3"), py::arg("T0"), 
            py::arg("rh_max_nh3"), py::arg("Jindex"), 
            py::arg("method") = "dry",
            py::arg("H2Oppmv") = 2500,
            py::arg("max_iter") = 200,
            "Return bottom temperature Ts for given T1bar.")

      .def(
          "get_rad",
          [](MeshBlock &mesh_block) { return mesh_block.pimpl->prad; },
          py::return_value_policy::reference)

      .def(
          "get_temp",
          [](MeshBlock &mesh_block, int k, int j, int i){ 
            auto pthermo = Thermodynamics::GetInstance();
            return pthermo->GetTemp(&mesh_block, k, j, i); },
          py::return_value_policy::reference)

      .def(
          "get_pressure",
          [](MeshBlock &mesh_block, int k, int j, int i){ 
            auto pthermo = Thermodynamics::GetInstance();
            return pthermo->GetPres(&mesh_block, k, j, i); },
          py::return_value_policy::reference)

      .def(
          "get_tracer",
          [](MeshBlock &mesh_block, int itracer, int k, int j, int i){ 
            auto ptracer = mesh_block.pimpl->ptracer;
            return ptracer->u(itracer, k, j, i); },
          py::return_value_policy::reference)

      // .def(
      //     "set_tracer",
      //     [](MeshBlock &mesh_block, int itracer, int k, int j, int i, double value){ 
      //       auto ptracer = mesh_block.pimpl->ptracer;
      //       ptracer->u(itracer, k, j, i)=value; })
      
      .def(
          "set_tracer_layer",
          [](MeshBlock &mesh_block, int itracer, int j, int i,  double value){ 
            return set_tracer_layer(&mesh_block, itracer, j, i, value); },
          py::arg("itracer"), py::arg("j"), py::arg("i"),  py::arg("value"),
          "Set tracer value for the given MeshBlock and Jindex.")

      .def(
          "get_theta",
          [](MeshBlock &mesh_block, Real p0, int k, int j, int i){ 
            auto pthermo = Thermodynamics::GetInstance();
            return pthermo->PotentialTemp(&mesh_block, p0, k, j, i); },
          py::return_value_policy::reference)

      .def("get_aircolumn",
          [](MeshBlock &mesh_block, int k, int j, int il, int iu){
            return AirParcelHelper::gather_from_primitive(&mesh_block, k, j, il, iu);
          },
          py::return_value_policy::reference)
      
      .def("distribute_to_primitive",
          [](MeshBlock &mesh_block, int k, int j, int i, AirParcel &airparcel){
            return AirParcelHelper::distribute_to_primitive(&mesh_block, k, j, i, airparcel);
          },
          py::return_value_policy::reference);

  // outputs
  py::class_<Outputs>(m, "Outputs")
      .def(py::init<Mesh *, ParameterInput *>(), py::arg("mesh"),
           py::arg("pin"))

      .def("make_outputs", &Outputs::MakeOutputs, py::arg("mesh"),
           py::arg("pin"), py::arg("wtflag") = false);
}
