#ifndef RLGAMES_RESNET_LAYER
#define RLGAMES_RESNET_LAYER

#include <vector>

#include <pytorch_util.h>

#include <torch/torch.h>

namespace rlgames {

namespace s = std;
namespace t = torch;

struct ConvResnetLayerOptions {
  uint      xcsz;
  TensorDim c1sz;
  TensorDim c2sz;

  ConvResnetLayerOptions() = default;
  ConvResnetLayerOptions(uint xcsz, TensorDim c1sz, TensorDim c2sz):
    xcsz(xcsz),
    c1sz(c1sz),
    c2sz(c2sz)
  {}
  ConvResnetLayerOptions(const s::vector<TensorDim> dims){
    assert(dims.size() == 3);

    xcsz = dims[0].i;
    c1sz = dims[1];
    c2sz = dims[2];
  }
};

class ConvResnetLayerV1Impl : public t::nn::Module {
  t::nn::Conv2d c1, c2;
  t::nn::BatchNorm2d b1, b2;
public:
  explicit ConvResnetLayerV1Impl(const ConvResnetLayerOptions& opt):
    c1(register_module("c1", t::nn::Conv2d(t::nn::Conv2dOptions(opt.xcsz, opt.c1sz.i, opt.c1sz.j).padding((opt.c1sz.j - 1) / 2)))),
    c2(register_module("c2", t::nn::Conv2d(t::nn::Conv2dOptions(opt.c1sz.i, opt.c2sz.i, opt.c2sz.j).padding((opt.c2sz.j - 1) / 2)))),
    b1(register_module("b1", t::nn::BatchNorm2d(opt.c1sz.i))),
    b2(register_module("b2", t::nn::BatchNorm2d(opt.c2sz.i)))
  {}
  t::Tensor forward(t::Tensor x){
    if (x.dim() == 3){
      uint i = x.size(0);
      uint j = x.size(1);
      uint k = x.size(2);
      x = x.reshape({1, i, j, k});
    }
    t::Tensor xt = t::relu(b1(c1(x)));
    xt = b2(c2(xt));
    xt = t::relu(xt + x);
    return xt;
  }
};
TORCH_MODULE(ConvResnetLayerV1);

class ConvResnetLayerV2Impl : public t::nn::Module {
  t::nn::Conv2d c1, c2;
  t::nn::BatchNorm b1, b2;
public:
  explicit ConvResnetLayerV2Impl(const ConvResnetLayerOptions& opt):
    c1(register_module("c1", t::nn::Conv2d(t::nn::Conv2dOptions(opt.xcsz, opt.c1sz.i, opt.c1sz.j).padding(opt.c1sz.j - 1)))),
    c2(register_module("c2", t::nn::Conv2d(t::nn::Conv2dOptions(opt.c1sz.i, opt.c2sz.i, opt.c2sz.j).padding(opt.c2sz.j - 1)))),
    b1(register_module("b1", t::nn::BatchNorm(opt.xcsz))),
    b2(register_module("b2", t::nn::BatchNorm(opt.c1sz.i)))
  {}
  t::Tensor forward(t::Tensor x){
    if (x.dim() == 3){
      uint i = x.size(0);
      uint j = x.size(1);
      uint k = x.size(2);
      x = x.reshape({1, i, j, k});
    }
    t::Tensor xt = c2(t::relu(b2(c1(t::relu(b1(x))))));
    xt = xt + x;
    return xt;
  }
};
TORCH_MODULE(ConvResnetLayerV2);

} // rlgames

#endif//RLGAMES_RESNET_LAYER
