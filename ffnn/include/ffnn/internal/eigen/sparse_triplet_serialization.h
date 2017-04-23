/**
 * @file  dense_base_addons.h
 * @brief Reference: https://gist.github.com/mtao/5798888
 */
#ifndef FFNN_CONFIG_EIGEN_TRIPLET_SERIALIZATION_H
#define FFNN_CONFIG_EIGEN_TRIPLET_SERIALIZATION_H

namespace Eigen
{
template<typename Scalar, typename StorageIndex>
class Triplet;
} // namespace Eigen

namespace boost
{
namespace serialization
{
template <class Archive, typename ScalarType, typename StorageIndex>
void save(Archive & ar, const Eigen::Triplet<ScalarType, StorageIndex> & m, const unsigned int version)
{
    ar & m.row();
    ar & m.col();
    ar & m.value();
}

template <class Archive, typename ScalarType, typename StorageIndex>
void load(Archive & ar, Eigen::Triplet<ScalarType, StorageIndex> & m, const unsigned int version)
{
    int row,col;
    ScalarType value;
    ar & row;
    ar & col;
    ar & value;
    m = Eigen::Triplet<ScalarType, StorageIndex>(row,col,value);
}

template <class Archive, typename ScalarType, typename StorageIndex>
void serialize(Archive & ar, Eigen::Triplet<ScalarType, StorageIndex> & m, const unsigned int version)
{
    boost::serialization::split_free(ar,m,version);
}
}  // namespace serialization
}  // namespace boost
#endif