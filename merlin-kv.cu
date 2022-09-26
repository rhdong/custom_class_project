// This header is all you need to do the C++ portions of this
// tutorial
#include <torch/script.h>
// This header is what defines the custom class registration
// behavior specifically. script.h already includes this, but
// we include it here so you know it exists in case you want
// to look at the API or implementation.
#include <torch/custom_class.h>
#include <string>
#include <vector>
#include "include/merlin_hashtable.cuh"

template <class K, class V, class M, size_t D>
struct MerlinHashTable : torch::CustomClassHolder {
 public:
  using Table = nv::merlin::HashTable<K, V, M, D>;
  using TableOptions = nv::merlin::HashTableOptions;

 public:
  MerlinHashTable() { table_ = std::make_unique<Table>(); }
  void init(int64_t init_capacity, int64_t max_capacity,
            int64_t max_hbm_for_vectors, double max_load_factor) {
    options_.init_capacity = init_capacity;
    options_.max_capacity = 8192;
    options_.max_hbm_for_vectors = 0;
    options_.io_by_cpu = false;
    options_.max_load_factor = static_cast<float>(max_load_factor);

    table_->init(options_);
  }

  void insert_or_assign(int64_t n, torch::Tensor keys, torch::Tensor values) {
    table_->insert_or_assign(static_cast<size_t>(n),
                             reinterpret_cast<K*>(keys.data_ptr<int64_t>()),
                             values.data_ptr<V>());
  }

  void find(int64_t n, torch::Tensor keys, torch::Tensor values,
            torch::Tensor found) {
    table_->find(static_cast<size_t>(n),
                 reinterpret_cast<K*>(keys.data_ptr<int64_t>()),
                 values.data_ptr<V>(), found.data_ptr<bool>());
  }

  int64_t size() { return static_cast<int64_t>(table_->size()); }

  int64_t capacity() { return static_cast<int64_t>(table_->capacity()); }

  void clear() { return table_->clear(); }

 public:
  std::unique_ptr<Table> table_;
  TableOptions options_;
};

TORCH_LIBRARY(merlin_kv, m) {
  m.class_<MerlinHashTable<uint64_t, float, uint64_t, 2>>("HashTable")
      .def(torch::init<>())
      .def("init", &MerlinHashTable<uint64_t, float, uint64_t, 2>::init)
      .def("insert_or_assign",
           &MerlinHashTable<uint64_t, float, uint64_t, 2>::insert_or_assign)
      .def("find", &MerlinHashTable<uint64_t, float, uint64_t, 2>::find)
      .def("size", &MerlinHashTable<uint64_t, float, uint64_t, 2>::size)
      .def("capacity", &MerlinHashTable<uint64_t, float, uint64_t, 2>::capacity)
      .def("clear", &MerlinHashTable<uint64_t, float, uint64_t, 2>::clear);
}