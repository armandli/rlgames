#ifndef GRIDWORLD_LEARNING_UTIL
#define GRIDWORLD_LEARNING_UTIL

#include <torch/torch.h>

#include <cassert>
#include <vector>
#include <random>

namespace gridworld_pt {

namespace t = torch;
namespace s = std;

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

struct Dim {
  uint x, y, z;
  Dim(): x(0), y(0), z(0) {}
  explicit Dim(uint x): x(x), y(0), z(0) {}
  Dim(uint x, uint y): x(x), y(y), z(0) {}
  Dim(uint x, uint y, uint z): x(x), y(y), z(z) {}
  Dim(const Dim& o):
    x(o.x), y(o.y), z(o.z)
  {}
  Dim& operator=(const Dim& o){
    x = o.x;
    y = o.y;
    z = o.z;
    return *this;
  }

  uint size() const {
    if (z > 0)      return 3U;
    else if (y > 0) return 2U;
    else if (x > 0) return 1U;
    else            return 0U;
  }

  uint flatten_size() const {
    if (z > 0)      return x * y * z;
    else if (y > 0) return x * y;
    else if (x > 0) return x;
    else            return 0U;
  }
};

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

} // gridworld_pt


#endif//GRIDWORLD_LEARNING_UTIL
