/**
 * @file  dense_base_addons.h
 * @brief Reference: https://gist.github.com/mtao/5798888
 */
#ifndef FFNN_CONFIG_EIGEN_SPARSE_MATRIX_BASE_ADDONS_H
#define FFNN_CONFIG_EIGEN_SPARSE_MATRIX_BASE_ADDONS_H

template <class Archive>
void save(Archive & ar, const unsigned int version) const
{
  using Triplet = typename Eigen::Triplet<Scalar>;
  Index inner_size = derived().innerSize();
  Index outer_size = derived().outerSize();

  std::vector<Triplet> triplets;
  triplets.reserve(inner_size * outer_size);
  for(Index idx = 0; idx < outer_size; idx++)
  {
    for(typename Eigen::SparseMatrix<Scalar>::InnerIterator it(derived(), idx); it; ++it)
    {
      triplets.emplace_back(it.row(), it.col(), it.value());
    }
  }

  ar & inner_size;
  ar & outer_size;
  ar & triplets;
}

template <class Archive>
void load(Archive& ar, const unsigned int version) 
{
  using Triplet = typename Eigen::Triplet<Scalar>;
  Index inner_size;
  Index outer_size;
  ar & inner_size;
  ar & outer_size;

  Index rows = derived().IsRowMajor ? outer_size : inner_size;
  Index cols = derived().IsRowMajor ? inner_size : outer_size;
  derived().resize(rows, cols);

  std::vector<Triplet> triplets;
  ar & triplets;
  derived().setFromTriplets(triplets.begin(), triplets.end());
}

template<class Archive>
void serialize(Archive & ar, const unsigned int file_version)
{
  boost::serialization::split_member(ar, *this, file_version);
}
#endif // FFNN_CONFIG_EIGEN_SPARSE_MATRIX_BASE_ADDONS_H
