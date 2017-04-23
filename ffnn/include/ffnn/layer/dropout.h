/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_DROPOUT_H
#define FFNN_LAYER_DROPOUT_H

// Boost
#include <boost/dynamic_bitset.hpp>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/layer/hidden.h>

namespace ffnn
{
namespace layer
{
/**
 * @brief Probabilistic dropout-modeling layer type
 */
template<typename ValueType, FFNN_SIZE_TYPE SizeAtCompileTime = Eigen::Dynamic>
class Dropout :
  public Hidden<ValueType, SizeAtCompileTime, SizeAtCompileTime>
{
public:
  /// Base type alias
  using Base = Hidden<ValueType, SizeAtCompileTime, SizeAtCompileTime>;

  /// Size type standardization
  typedef typename Base::SizeType SizeType;

  /// Offset type standardization
  typedef typename Base::OffsetType OffsetType;

  // Probability (real-number) type standardization
  typedef ValueType ProbabilityType;

  /// A configuration object for a Dropout layer
  struct Parameters
  {
    /// Probability of a unit dropout (uniform)
    ProbabilityType probability;

    /**
     * @brief Selects propagation behavior
     *
     *        - if <code>blind == true</code>, then all error values
     *          are propagated backwards as if there was not dropout
     *        - if <code>blind == false</code>, then only errors
     *          of nodes which were not disconnected are propagated backwards
     */
    bool blind;

    /// Default constructor
    Parameters(const ProbabilityType& probability = 0.5, bool blind = true);
  };

  /**
   * @brief Default constructor
   * @param config  layer configuration object
   */
  Dropout(const Parameters& config = Parameters());
  virtual ~Dropout();

  /**
   * @brief Initialize the layer
   */
  virtual bool initialize();

  /**
   * @brief Forward value propagation
   * @retval true  if forward-propagation succeeded
   * @retval false  otherwise
   */
  virtual bool forward();

  /**
   * @brief Backward value propagation
   * @retval true  if backward-propagation succeeded
   * @retval false  otherwise
   */
  virtual bool backward();

private:
  /// Layer configuration
  Parameters config_;

  /// Flags for connectedness under dropout
  boost::dynamic_bitset<> connected_;
};
}  // namespace layer
}  // namespace ffnn

/// FFNN (implementation)
#include <ffnn/layer/impl/dropout.hpp>
#endif  // FFNN_LAYER_DROPOUT_H
