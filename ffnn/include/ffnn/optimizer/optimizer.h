/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_OPTIMIZER_OPTIMIZER_H
#define FFNN_LAYER_OPTIMIZER_OPTIMIZER_H

// C++ Standard Library
#include <string>
#include <type_traits>

// Boost
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

// FFNN
#include <ffnn/internal/config.h>
#include <ffnn/assert.h>
#include <ffnn/optimizer/loss_function.h>

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

  /// Scalar type standardization
  typedef typename LayerType::Scalar Scalar;

  /**
   * @brief Naming constructor
   * @param name  name associated with the optimizer
   */
  explicit
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
 * @param opt Optimizer type
 * @note An optimizer must be registered to a Layer to allow use
 */
#define FFNN_REGISTER_OPTIMIZER(opt)\
  template<typename _LayerType, ::ffnn::optimizer::LossFunction _LossFunctionSpec>\
  friend class ::ffnn::optimizer::opt;\
  template<typename _LayerType, ::ffnn::optimizer::LossFunction _LossFunctionSpec>\
  friend class ::ffnn::optimizer::opt##_;

}  // namespace optimizer
}  // namespace ffnn
#endif  // FFNN_LAYER_OPTIMIZER_OPTIMIZER_H
