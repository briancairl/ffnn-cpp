/**
 * @note HEADER-ONLY IMPLEMENTATION FILE
 * @warn Do not include directly
 */

// FFNN
#include <ffnn/assert.h>
#include <ffnn/logging.h>
#include <ffnn/internal/signature.h>

namespace ffnn
{
namespace layer
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
};

#define HIDDEN_PARAMS ValueType, InputsHeightAtCompileTime, InputsWidthAtCompileTime, OutputsHeightAtCompileTime, OutputsWidthAtCompileTime
#define HIDDEN_PARAMS_ADVANCED _InputBlockType, _OutputBlockType, _InputMappingType, _OutputMappingType
#define HIDDEN Hidden<HIDDEN_PARAMS, HIDDEN_PARAMS_ADVANCED>

template<typename ValueType,
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
HIDDEN::Hidden(const DimensionsType& input_dim,
               const DimensionsType& output_dim) :
  Base(input_dim.size(), output_dim.size()),
  input_dim_(input_dim),
  output_dim_(output_dim)
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
HIDDEN::~Hidden()
{}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
typename HIDDEN::OffsetType 
HIDDEN::connectToForwardLayer(const Base& next, OffsetType offset)
{
  // Map output of next layer to input buffer
  auto output_ptr = const_cast<ValueType*>(next.getInputBuffer().data()) + offset;
  output_ = _OutputMappingType::create(output_ptr,
                                       output_dim_.height,
                                       output_dim_.width);

  // Map error of next layer to backward-error buffer
  auto error_ptr = const_cast<ValueType*>(next.getBackwardErrorBuffer().data()) + offset;
  forward_error_ = _OutputMappingType::create(error_ptr,
                                              output_dim_.height,
                                              output_dim_.width);

  // Return next offset after assigning buffer segments
  return offset + Base::outputSize();
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
bool HIDDEN::initialize()
{
  // Abort if layer is already initialized
  if (!Base::setupRequired() && Base::isInitialized())
  {
    FFNN_WARN_NAMED("layer::Hidden", "<" << Base::getID() << "> already initialized.");
    return false;
  }

  // Resolve input dimensions from previous layer output dimensions
  Base::input_size_ = Base::evaluateInputSize();

  // Validate input count
  FFNN_STATIC_ASSERT_MSG (input_dim_.size() == Base::inputSize(),
                          "Specified input size is incompatible with expected input dimensions.");

  // Validate output count
  FFNN_STATIC_ASSERT_MSG (output_dim_.size() == Base::outputSize(),
                          "Specified output size is incompatible with expected output dimensions.");

  // Do basic initialization
  if (Base::initialize())
  {
    // Create input buffer map
    auto input_ptr = const_cast<ValueType*>(Base::getInputBuffer().data());
    input_ = _InputMappingType::create(input_ptr,
                                       input_dim_.height,
                                       input_dim_.width);

    // Create input buffer map
    auto error_ptr = const_cast<ValueType*>(Base::getBackwardErrorBuffer().data()),
    backward_error_ = _InputMappingType::create(error_ptr,
                                                input_dim_.height,
                                                input_dim_.width);

    FFNN_DEBUG_NAMED("layer::Hidden", "Created forward mappings.");

    // Resolve previous layer output buffers
    if (Base::connectInputLayers() == Base::inputSize())
    {
      FFNN_DEBUG_NAMED("layer::Hidden",
                       "<" <<
                       Base::getID() <<
                       "> initialized as (in=" <<
                       Base::inputSize()  <<
                       ", out=" <<
                       Base::outputSize() <<
                       ")");
      return Base::isInitialized();
    }

    // Initialization failed
    Base::initialized_ = false;
    FFNN_ERROR_NAMED("layer::Hidden", "<" << Base::getID() << "> bad input count after input resolution.");
  }
  // Error initializing
  FFNN_ERROR_NAMED("layer::Hidden", "<" << Base::getID() << "> failed to initialize.");
  return false;
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
void HIDDEN::save(typename HIDDEN::OutputArchive& ar,
                            typename HIDDEN::VersionType version) const
{
  ffnn::io::signature::apply<HIDDEN>(ar);
  Base::save(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Saved");
}

template<typename ValueType,
         FFNN_SIZE_TYPE InputsHeightAtCompileTime,
         FFNN_SIZE_TYPE InputsWidthAtCompileTime,
         FFNN_SIZE_TYPE OutputsHeightAtCompileTime,
         FFNN_SIZE_TYPE OutputsWidthAtCompileTime,
         typename _InputBlockType,
         typename _OutputBlockType,
         typename _InputMappingType,
         typename _OutputMappingType>
void HIDDEN::load(typename HIDDEN::InputArchive& ar,
                            typename HIDDEN::VersionType version)
{
  ffnn::io::signature::check<HIDDEN>(ar);
  Base::load(ar, version);
  FFNN_DEBUG_NAMED("layer::Hidden", "Loaded");
}

#undef HIDDEN
}  // namespace layer
}  // namespace ffnn
