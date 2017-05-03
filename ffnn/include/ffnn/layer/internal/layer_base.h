/**
 * @author Brian Cairl
 * @date 2017
 */
#ifndef FFNN_LAYER_INTERNAL_FORWARD_INTERFACE_H
#define FFNN_LAYER_INTERNAL_FORWARD_INTERFACE_H

// FFNN (internal)
#include <ffnn/internal/traits/serializable.h>
#include <ffnn/internal/traits/unique.h>
#include <ffnn/internal/signature.h>

// FFNN
#include <ffnn/config/global.h>

namespace ffnn
{
namespace layer
{
namespace internal
{
#define IS_DYNAMIC(x) (x == Eigen::Dynamic)
#define IS_DYNAMIC_PAIR(n, m) (IS_DYNAMIC(n) || IS_DYNAMIC(m))
#define IS_DYNAMIC_TRIPLET(n, m, l) (IS_DYNAMIC(n) || IS_DYNAMIC(m) || IS_DYNAMIC(l))
#define PROD_IF_STATIC_PAIR(n, m) (IS_DYNAMIC_PAIR(n, m) ? Eigen::Dynamic : (n*m))
#define PROD_IF_STATIC_TRIPLET(n, m, l) (IS_DYNAMIC_TRIPLET(n, m, l) ? Eigen::Dynamic : (n*m*l))



template<typename SizeType>
struct Dimensions
{
  SizeType height;
  SizeType width;
  SizeType depth;

  Dimensions() :
    height(Eigen::Dynamic),
    width(Eigen::Dynamic),
    depth(Eigen::Dynamic)
  {}

  explicit
  Dimensions(SizeType height, SizeType width = 1, SizeType depth = 1) :
    height(height),
    width(width),
    depth(depth)
  {}

  inline SizeType size() const
  {
    return PROD_IF_STATIC_TRIPLET(height, width, depth);
  }

  inline bool valid() const
  {
    return PROD_IF_STATIC_TRIPLET(height, width, depth) > 0;
  }

  operator SizeType() const { return size(); }

  void operator=(SizeType count)
  {
    height = count;
    width = 1;
    depth = 1;
  }

  void operator=(const Dimensions& dim)
  {
    height = dim.height;
    width = dim.width;
    depth = dim.depth;
  }
};

template<typename SizeType>
std::ostream& operator<<(std::ostream& os, const Dimensions<SizeType>& dim)
{
  os << "<" << dim.height << " x " << dim.width << " x " << dim.depth << ">";
  return os;
}

/**
 * @brief Base object for all layer types
 */
template<typename ValueType>
class LayerBase :
  public traits::Unique
{
public:
  /// Scalar type standardization
  typedef ValueType ScalarType;

  /// Size type standardization
  typedef FFNN_SIZE_TYPE SizeType;

  /// Offset type standardization
  typedef FFNN_OFFSET_TYPE OffsetType;

  /// Dimension type standardization
  typedef Dimensions<SizeType> DimType;

  explicit
  LayerBase(const DimType& input_dim  = DimType(Eigen::Dynamic),
            const DimType& output_dim = DimType(Eigen::Dynamic)) :
    initialized_(false),
    setup_required_(true),
    input_dim_(input_dim),
    output_dim_(output_dim)
  {}
  virtual ~LayerBase() {}

  /**
   * @brief Returns the total number of Layer inputs
   */
  inline SizeType inputSize() const
  {
    return input_dim_.size();
  }

  /**
   * @brief Returns the total number of Layer outputs
   */
  inline SizeType outputSize() const
  {
    return output_dim_.size();
  }

  /**
   * @brief Returns the total number counted (evaluated) inputs
   */
  virtual SizeType evaluateInputSize() const
  {
    return input_dim_.size();
  }

  /**
   * @brief Initialize the layer
   */
  virtual bool initialize() = 0;

  /**
   * @brief Returns true if layer has been initialized
   * @retval true  if layer is initialized
   * @retval false  otherwise
   */
  virtual bool isInitialized() const
  {
    return initialized_;
  }

  /**
   * @brief Returns true if portions of the interface must be setup
   * @retval true  if reinitialization allowed
   * @retval false  otherwise
   */
  bool setupRequired() const
  {
    return setup_required_;
  }

protected:
  FFNN_REGISTER_SERIALIZABLE(LayerBase)

  /// Save serializer
  void save(OutputArchive& ar, VersionType version) const
  {
    ffnn::io::signature::apply<LayerBase<ValueType>>(ar);
    traits::Unique::save(ar, version);

    // Load flags
    ar & initialized_;

    // Load dimensions
    ar & input_dim_.height;
    ar & input_dim_.width;
    ar & input_dim_.depth;

    ar & output_dim_.height;
    ar & output_dim_.width;
    ar & output_dim_.depth;
  }

  /// Load serializer
  void load(InputArchive& ar, VersionType version)
  {
    ffnn::io::signature::check<LayerBase<ValueType>>(ar);
    traits::Unique::load(ar, version);

    // Load flags
    ar & initialized_;

    // Load dimensions
    ar & input_dim_.height;
    ar & input_dim_.width;
    ar & input_dim_.depth;

    ar & output_dim_.height;
    ar & output_dim_.width;
    ar & output_dim_.depth;

    setup_required_ = false;
  }

  /// Flags which indicated that layer has been initialized
  bool initialized_;

  /// Flags which indicates that layer should be initialized normally
  bool setup_required_;

  /// Total number of input connections
  DimType input_dim_;

  /// Total number of output connections
  DimType output_dim_;
};
}  // namespace internal
}  // namespace layer
}  // namespace ffnn
#endif  // FFNN_LAYER_INTERNAL_FORWARD_INTERFACE_H
