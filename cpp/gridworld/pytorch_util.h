#include <torch/torch.h>

#include <cassert>
#include <vector>

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

} // gridworld_pt
