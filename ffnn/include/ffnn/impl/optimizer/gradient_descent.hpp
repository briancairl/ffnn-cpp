template<>
template <typename ValueType,
          typename Options,
          typename Extrinsics>
class GradientDescent<layer::FullyConnected<ValueType, Options, Extrinsics>>:
  public Optimizer<layer::FullyConnected<ValueType, Options, Extrinsics>>
{
public:
  /// Matrix type standardization
  typedef typename LayerType::InputBlockType InputBlockType;

  /// Matrix type standardization
  typedef typename LayerType::ParametersType ParametersType;

  /**
   * @brief Setup constructor
   * @param lr  Learning rate
   */
  explicit
  GradientDescent(ValueType lr) :
    Optimizer<LayerType>("GradientDescent[FullyConnected]"),
    lr_(lr)
  {}
  virtual ~GradientDescent() {}

  /**
   * @brief Initializes the Optimizer
   * @param[in, out] layer  Layer to optimize
   */
  virtual void initialize(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Capture gradient sizing
    gradient_ = layer.parameters_;

    // Capture input sizing
    prev_input_ = layer.input_;

    // Reset optimizer
    reset(layer);
  }

  /**
   * @brief Resets persistent Optimizer states
   * @param[in, out] layer  Layer to optimize
   */
  virtual void reset(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Reset gradient
    gradient_.setZero();
  }

  /**
   * @brief Computes one forward optimization update step
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization setp was successful
   * @retval false  otherwise
   */
  virtual bool forward(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Copy current input for updating
    prev_input_.noalias() = layer.input_;
    return true;
  }

  /**
   * @brief Computes optimization step during backward propogation
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization setp was successful
   * @retval false  otherwise
   */
  virtual bool backward(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Compute and accumulate new gradient
    gradient_.weights.noalias() += layer.forward_error_ * prev_input_.transpose();
    gradient_.biases.noalias() += layer.forward_error_;

    // Back-prop error
    return true;
  }

  /**
   * @brief Applies optimization update
   * @param[in, out] layer  Layer to optimize
   * @retval true  if optimization update was applied successfully
   * @retval false  otherwise
   */
  virtual bool update(LayerType& layer)
  {
    FFNN_ASSERT_MSG(layer.isInitialized(), "Layer to optimize is not initialized.");

    // Incorporate learning rate
    gradient_ *= lr_;

    // Update parameters
    layer.parameters_ -= gradient_;

    // Reinitialize optimizer
    reset(layer);
    return true;
  }

protected:
  /// Learning rate
  ValueType lr_;

  /// Previous input
  InputBlockType prev_input_;

  /// Coefficient gradient
  ParametersType gradient_;
};
}  // namespace optimizer
}  // namespace ffnn