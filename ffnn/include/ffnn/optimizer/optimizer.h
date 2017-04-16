/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OPTIMIZATION_OPTIMIZER_H
#define FFNN_LAYER_OPTIMIZATION_OPTIMIZER_H

// C++ Standard Library
#include <string>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/assert.h>
#include <ffnn/traits/shared.h>

namespace ffnn
{
namespace optimizer
{
/**
 * @brief A layer-wise optimizer visitor
 */
template<typename LayerType>
class Optimizer :
  public traits::Shared<Optimizer<LayerType>>
{
public:
  Optimizer(const std::string& name) :
    name_(name)
  {}

  /**
   * @brief Initializes the Optimizer
   * @param[in, out] layer  Layer to optimize
   */
  virtual void initialize(LayerType& layer) = 0;

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

private:
  /// Name of the optimizer
  const std::string name_;
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
