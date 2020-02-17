#ifndef RLGAMES_PYTORCH_UTIL
#define RLGAMES_PYTORCH_UTIL

#include <cassert>
#include <vector>
#include <random>

#include <type_alias.h>

#include <torch/torch.h>

namespace rlgames {

namespace s = std;
namespace t = torch;

template <typename MODULE>
void copy_state(MODULE dst, MODULE src){
  t::autograd::GradMode::set_enabled(false);

  s::vector<t::Tensor> src_params = src->parameters(true);
  s::vector<t::Tensor> dst_params = dst->parameters(true);

  assert(src_params.size() == dst_params.size());

  for (size_t i = 0; i < src_params.size(); ++i)
    dst_params[i].copy_(src_params[i]);

  s::vector<t::Tensor> src_buffers = src->buffers(true);
  s::vector<t::Tensor> dst_buffers = dst->buffers(true);

  assert(src_buffers.size() == dst_buffers.size());

  for (size_t i = 0; i < src_buffers.size(); ++i)
    dst_buffers[i].copy_(src_buffers[i]);

  t::autograd::GradMode::set_enabled(true);
}

template <typename ENG>
uint sample_discrete_distribution(float* probabilities, size_t size, ENG& eng){
  s::discrete_distribution<uint> dist(probabilities, probabilities + size);
  return dist(eng);
}

//helper function that creates a ConvOptions<D> from ConvNdOptions<D>, which does not exist in pytorch
//NOTE: this loses information regarding
template <size_t D>
t::nn::ConvOptions<D> conv_options(const t::nn::detail::ConvNdOptions<D>& convnd_options){
  return t::nn::ConvOptions<D>(convnd_options.in_channels(), convnd_options.out_channels(), convnd_options.kernel_size())
    .stride(convnd_options.stride())
    .padding(convnd_options.padding())
    .dilation(convnd_options.dilation())
    .groups(convnd_options.groups())
    .bias(convnd_options.bias())
    .padding_mode(convnd_options.padding_mode())
  ;
}

struct TensorDim {
  uint i,j,k;

  TensorDim() = default;
  explicit TensorDim(uint i): i(i), j(0U), k(0U) {}
  TensorDim(uint i, uint j): i(i), j(j), k(0U) {}
  TensorDim(uint i, uint j, uint k): i(i), j(j), k(k) {}

  uint dim() const {
    if      (i == 0U) return 0U;
    else if (j == 0U) return 1U;
    else if (k == 0U) return 2U;
    else              return 3U;
  }

  uint flatten_size() const {
    if      (i == 0U) return 0U;
    else if (j == 0U) return i;
    else if (k == 0U) return i * j;
    else              return i * j * k;
  }
};

//pair of TensorDim
struct TensorDimP {
  TensorDim x, y;
  TensorDimP() = default;
  TensorDimP(TensorDim&& x, TensorDim&& y): x(s::move(x)), y(s::move(y)) {}
};

//pair of tensors
struct TensorP {
  t::Tensor x, y;
  TensorP() = default;
  TensorP(t::Tensor x, t::Tensor y): x(x), y(y) {}
};

} // rlgames

#endif//RLGAMES_PYTORCH_UTIL
