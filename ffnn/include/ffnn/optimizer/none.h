/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OPTIMIZER_NONE_H
#define FFNN_LAYER_OPTIMIZER_NONE_H

// C++ Standard Library
#include <exception>
#include <string>

// FFNN
#include <ffnn/config/global.h>
#include <ffnn/assert.h>
#include <ffnn/optimizer/optimizer.h>

namespace ffnn
{
namespace optimizer
{
template<typename LayerType>
class None :
  public Optimizer<LayerType>
{
public:
  /**
   * @brief Default constructor
   */
  None() :
    Optimizer<LayerType>("None")
  {}

  /**
   * @brief Passthrough
   * @retval true
   */
  virtual void initialize(LayerType& layer)
  {}

  /**
   * @brief Passthrough
   * @retval true
   */
  virtual bool forward(LayerType& layer)
  {
    return true;
  }

  /**
   * @brief Does nothing
   * @warning This method will throw if called
   */
  virtual bool backward(LayerType& layer)
  {
    throw std::runtime_error("Optimizer::backward called on \"None\" optimizer.");
    return false;
  }

  /**
   * @brief Does nothing
   * @warning This method will throw if called
   */
  virtual bool update(LayerType& layer)
  {
    throw std::runtime_error("Optimizer::update called on \"None\" optimizer.");
    return false;
  }
};
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_OPTIMIZER_NONE_H