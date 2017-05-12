/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OPTIMIZATION_OPTIMIZER_H
#define FFNN_LAYER_OPTIMIZATION_OPTIMIZER_H

// C++ Standard Library
#include <string>
#include <type_traits>

// Boost
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/assert.h>

namespace ffnn
{
namespace optimizer
{
/**
 * @brief A layer-wise optimizer visitor
 */
template<typename LayerType>
class Optimizer
{
public:
  /// Shared resource standardization
  typedef boost::shared_ptr<Optimizer> Ptr;

  /// Constant shared resource standardization
  typedef boost::shared_ptr<const Optimizer> ConstPtr;

  /**
   * @brief Naming constructor
   * @param name  name associated with the optimizer
   */
  Optimizer(const std::string& name) :
    name_(name)
  {}
  virtual ~Optimizer() {}

  /**
   * @brief Initializes the Optimizer
   * @param[in, out] layer  Layer to optimize
   */
  virtual void initialize(LayerType& layer) = 0;

  /**
   * @brief Resetrs persistent Optimizer states
   * @param[in, out] layer  Layer to optimize
   */
  virtual void reset(LayerType& layer) = 0;

  /**
   * @brief Computes one optimizer update step
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimizer setp was successful
   * @retval false  otherwise
   */
  virtual bool forward(LayerType& layer) = 0;

  /**
   * @brief Computes one optimizer update step
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimizer setp was successful
   * @retval false  otherwise
   */
  virtual bool backward(LayerType& layer) = 0;

  /**
   * @brief Applies optimizer update
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimizer update was applied successfully
   * @retval false  otherwise
   */
  virtual bool update(LayerType& layer) = 0;

  /**
   * @brief Exposes name of the optimizer
   */
  inline const std::string& name() const
  {
    return name_;
  }

  /**
   * @brief Sets name of the optimizer
   */
  inline void setName(const std::string& name)
  {
    name_ = name;
  }

private:
  /// Name of the optimizer
  std::string name_;
};

/**
 * @brief Registers an optimizer to a Layer
 * @param layer layer type name
 * @param opt Optimizer type
 * @note An optimizer must be registered to a Layer to allow use
 */
#define FFNN_REGISTER_OPTIMIZER(layer, opt)\
  friend class ::ffnn::optimizer::opt<layer>;

}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_LAYER_OPTIMIZATION_OPTIMIZER_H
