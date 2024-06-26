#ifndef SRC_INDEX_MAP_HPP_
#define SRC_INDEX_MAP_HPP_

// C/C++
#include <map>
#include <string>

class ParameterInput;

class IndexMap {
 protected:
  //! Protected ctor access thru static member function Instance
  IndexMap() {}

 public:
  ~IndexMap();

  static IndexMap const* GetInstance();

  static IndexMap const* InitFromAthenaInput(ParameterInput* pin);

  static IndexMap const* InitFromNames(std::vector<std::string> const& vapors,
                                       std::vector<std::string> const& clouds,
                                       std::vector<std::string> const& tracers);

  static void Destroy();

  bool HasVapor(std::string const& name) const {
    return vapor_index_map_.find(name) != vapor_index_map_.end();
  }

  size_t GetVaporId(std::string const& name) const {
    return vapor_index_map_.at(name);
  }

  std::string GetVaporName(size_t i) const;

  bool HasCloud(std::string const& name) const {
    return cloud_index_map_.find(name) != cloud_index_map_.end();
  }

  size_t GetCloudId(std::string const& name) const {
    return cloud_index_map_.at(name);
  }

  std::string GetCloudName(size_t i) const;

  bool HasChemistry(std::string const& name) const {
    return chemistry_index_map_.find(name) != chemistry_index_map_.end();
  }

  size_t GetChemistryId(std::string const& name) const {
    return chemistry_index_map_.at(name);
  }

  bool HasTracer(std::string const& name) const {
    return tracer_index_map_.find(name) != tracer_index_map_.end();
  }

  size_t GetTracerId(std::string const& name) const {
    return tracer_index_map_.at(name);
  }

  std::string GetTracerName(size_t i) const;

  size_t GetSpeciesId(std::string category_name) const;

 private:
  std::map<std::string, size_t> vapor_index_map_;
  std::map<std::string, size_t> cloud_index_map_;
  std::map<std::string, size_t> chemistry_index_map_;
  std::map<std::string, size_t> tracer_index_map_;
  std::map<std::string, size_t> particle_index_map_;

  //! Pointer to the single IndexMap instance
  static IndexMap* myindex_map_;
};

#endif  // SRC_INDEX_MAP_HPP_
