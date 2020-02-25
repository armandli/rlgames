#ifndef RLGAMES_ZERO_MODEL_RESNET_SMALL
#define RLGAMES_ZERO_MODEL_RESNET_SMALL

#include <vector>

#include <models/resnet_layer.h>
#include <encoders/go_zero_encoder.h>
#include <models/model_base.h>

#include <torch/torch.h>

namespace rlgames {

struct ZeroModelResnetSmallOptions {
  TensorDim              c1sz;
  TensorDim              c2sz;
  TensorDim              c3sz;
  TensorDim              c4sz;
  TensorDim              pc1sz;
  TensorDim              vc1sz;
  ConvResnetLayerOptions c1rsz;
  ConvResnetLayerOptions c2rsz;
  ConvResnetLayerOptions c3rsz;
  ConvResnetLayerOptions c4rsz;

  ZeroModelResnetSmallOptions() = default;
  ZeroModelResnetSmallOptions(const s::vector<TensorDim> dims){
    assert(dims.size() == 18);

    c1sz = dims[0];
    c2sz = dims[1];
    c3sz = dims[2];
    c4sz = dims[3];
    pc1sz = dims[4];
    vc1sz = dims[5];
    c1rsz.xcsz = dims[6].i;
    c1rsz.c1sz = dims[7];
    c1rsz.c2sz = dims[8];
    c2rsz.xcsz = dims[9].i;
    c2rsz.c1sz = dims[10];
    c2rsz.c2sz = dims[11];
    c3rsz.xcsz = dims[12].i;
    c3rsz.c1sz = dims[13];
    c3rsz.c2sz = dims[14];
    c4rsz.xcsz = dims[15].i;
    c4rsz.c1sz = dims[16];
    c4rsz.c2sz = dims[17];
  }
};

class ZeroModelResnetSmallImpl : public t::nn::Module {
  ConvResnetLayerV1 c1r1, c1r2, c1r3, c1r4,
                    c2r1, c2r2, c2r3, c2r4,
                    c3r1, c3r2, c3r3, c3r4,
                    c4r1, c4r2, c4r3, c4r4;
  t::nn::Conv2d c1, c2, c3, c4, pc1, vc1;
  t::nn::Linear pl1, vl1;
  TensorDimP state_size;

  t::Tensor featurize_state(t::Tensor state){
    if (state.dim() == 1)
      state = state.reshape({1, state_size.y.i});
    return state;
  }
  t::Tensor featurize_board(t::Tensor board){
    if (board.dim() == 3)
      board = board.reshape({1, state_size.x.i, state_size.x.j, state_size.x.k});
    board = c1r4(c1r3(c1r2(c1r1(t::relu(c1(board))))));
    board = c2r4(c2r3(c2r2(c2r1(t::relu(c2(board))))));
    board = c3r4(c3r3(c3r2(c3r1(t::relu(c3(board))))));
    board = c4r4(c4r3(c4r2(c4r1(t::relu(c4(board))))));
    return board;
  }
public:
  explicit ZeroModelResnetSmallImpl(TensorDimP ssz, uint asz, const ZeroModelResnetSmallOptions& opt):
    c1r1(register_module("c1r1", ConvResnetLayerV1(opt.c1rsz))),
    c1r2(register_module("c1r2", ConvResnetLayerV1(opt.c1rsz))),
    c1r3(register_module("c1r3", ConvResnetLayerV1(opt.c1rsz))),
    c1r4(register_module("c1r4", ConvResnetLayerV1(opt.c1rsz))),
    c2r1(register_module("c2r1", ConvResnetLayerV1(opt.c2rsz))),
    c2r2(register_module("c2r2", ConvResnetLayerV1(opt.c2rsz))),
    c2r3(register_module("c2r3", ConvResnetLayerV1(opt.c2rsz))),
    c2r4(register_module("c2r4", ConvResnetLayerV1(opt.c2rsz))),
    c3r1(register_module("c3r1", ConvResnetLayerV1(opt.c3rsz))),
    c3r2(register_module("c3r2", ConvResnetLayerV1(opt.c3rsz))),
    c3r3(register_module("c3r3", ConvResnetLayerV1(opt.c3rsz))),
    c3r4(register_module("c3r4", ConvResnetLayerV1(opt.c3rsz))),
    c4r1(register_module("c4r1", ConvResnetLayerV1(opt.c4rsz))),
    c4r2(register_module("c4r2", ConvResnetLayerV1(opt.c4rsz))),
    c4r3(register_module("c4r3", ConvResnetLayerV1(opt.c4rsz))),
    c4r4(register_module("c4r4", ConvResnetLayerV1(opt.c4rsz))),
    c1(register_module("c1", t::nn::Conv2d(t::nn::Conv2dOptions(ssz.x.i, opt.c1sz.i, opt.c1sz.j)))),
    c2(register_module("c2", t::nn::Conv2d(t::nn::Conv2dOptions(opt.c1sz.i, opt.c2sz.i, opt.c2sz.j)))),
    c3(register_module("c3", t::nn::Conv2d(t::nn::Conv2dOptions(opt.c2sz.i, opt.c3sz.i, opt.c3sz.j)))),
    c4(register_module("c4", t::nn::Conv2d(t::nn::Conv2dOptions(opt.c3sz.i, opt.c4sz.i, opt.c4sz.j)))),
    pc1(register_module("pc1", t::nn::Conv2d(t::nn::Conv2dOptions(opt.c4sz.i, opt.pc1sz.i, opt.pc1sz.j)))),
    vc1(register_module("vc1", t::nn::Conv2d(t::nn::Conv2dOptions(opt.c4sz.i, opt.vc1sz.i, opt.vc1sz.j)))),
    pl1(register_module("pl1",
      t::nn::Linear(
        (ssz.x.j - opt.c1sz.j - opt.c2sz.j - opt.c3sz.j - opt.c4sz.j - opt.pc1sz.j + 5) *
        (ssz.x.k - opt.c1sz.k - opt.c2sz.k - opt.c3sz.k - opt.c4sz.k - opt.pc1sz.k + 5) *
        opt.pc1sz.i + ssz.y.i,
      asz))),
    vl1(register_module("vl1",
      t::nn::Linear(
        (ssz.x.j - opt.c1sz.j - opt.c2sz.j - opt.c3sz.j - opt.c4sz.j - opt.vc1sz.j + 5) *
        (ssz.x.k - opt.c1sz.k - opt.c2sz.k - opt.c3sz.k - opt.c4sz.k - opt.vc1sz.k + 5) *
        opt.vc1sz.i + ssz.y.i,
      1))),
    state_size(ssz)
  {}
  TensorP forward(TensorP state){
    state.x = featurize_board(state.x);
    state.y = featurize_state(state.y);
    t::Tensor q = t::relu(pc1(state.x));
    q = t::softmax(pl1(t::cat({q.flatten(1, -1), state.y}, -1)), -1);
    t::Tensor v = t::relu(vc1(state.x));
    v = t::tanh(vl1(t::cat({v.flatten(1, -1), state.y}, -1)));
    return TensorP(q.squeeze(), v.squeeze());
  }
};
TORCH_MODULE(ZeroModelResnetSmall);

} // rlgames

#endif//RLGAMES_ZERO_MODEL_RESNET_SMALL
