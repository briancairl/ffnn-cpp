/**
 * @file  dense_base_addons.h
 * @brief http://stackoverflow.com/questions/18382457/eigen-and-boostserialize
 */
#ifndef FFNN_CONFIG_EIGEN_DENSE_BASE_ADDONS_H
#define FFNN_CONFIG_EIGEN_DENSE_BASE_ADDONS_H

template<class Archive>
void save(Archive & ar, const unsigned int version) const
{
  derived().eval();
  const Index rows = derived().rows();
  const Index cols = derived().cols();
  ar & rows;
  ar & cols;
  for (Index jdx = 0; jdx < cols; jdx++)
  {
    for (Index idx = 0; idx < rows; idx++)
    {
      ar & derived().coeff(idx, jdx);
    }
  }
}

template<class Archive>
void load(Archive & ar, const unsigned int version)
{
  Index rows, cols;
  ar & rows;
  ar & cols;
  if (rows != derived().rows() || cols != derived().cols())
  {
    derived().resize(rows, cols);
  }
  ar & boost::serialization::make_array(derived().data(), derived().size());
}

template<class Archive>
void serialize(Archive & ar, const unsigned int file_version)
{
  boost::serialization::split_member(ar, *this, file_version);
}

#endif // FFNN_CONFIG_EIGEN_DENSE_BASE_ADDONS_H