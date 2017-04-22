/**
 * @file  dense_base_addons.h
 * @brief Reference: https://gist.github.com/mtao/5798888
 */
#ifndef FFNN_CONFIG_EIGEN_SPARSE_MATRIX_BASE_ADDONS_H
#define FFNN_CONFIG_EIGEN_SPARSE_MATRIX_BASE_ADDONS_H

template <class Archive>
void save(Archive & ar, const unsigned int version) const
{
  using InnerIterator = typename Eigen::SparseMatrix<Scalar>::InnerIterator;
  Index innerSize = this->innerSize();
  Index outerSize = this->outerSize();

  ar & innerSize;
  ar & outerSize;
  for(Index idx = 0; idx < outerSize; idx++)
  {
    for(InnerIterator it(*this, idx); it; ++it)
    {
      Index row(it.row());
      Index col(it.col());
      Scalar value(it.value());

      ar & row;
      ar & col;
      ar & value;
    }
  }
}

template <class Archive>
void load(Archive& ar, const unsigned int version) 
{
  using Triplet = typename Eigen::Triplet<Scalar>;
  Index innerSize;
  Index outerSize;
  ar & innerSize;
  ar & outerSize;

  Index rows = this->IsRowMajor ? outerSize : innerSize;
  Index cols = this->IsRowMajor ? innerSize : outerSize;
  this->resize(rows, cols);

  std::vector<Triplet> triplets;
  triplets.reserve(innerSize * outerSize);
  for(Index idx = 0; idx < outerSize; idx++)
  {
    for(Index jdx = 0; jdx < innerSize; jdx++)
    {
      Index row;
      Index col;
      Scalar value;

      ar & row;
      ar & col;
      ar & value;
      triplets.emplace_back(row, col, value);
    }
  }
  this->setFromTriplets(triplets.begin(), triplets.end());
}

template<class Archive>
void serialize(Archive & ar, const unsigned int file_version)
{
  boost::serialization::split_member(ar, *this, file_version);
}
#endif // FFNN_CONFIG_EIGEN_SPARSE_MATRIX_BASE_ADDONS_H
