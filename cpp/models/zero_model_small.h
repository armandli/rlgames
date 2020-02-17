#ifndef RLGAMES_ZERO_MODEL
#define RLGAMES_ZERO_MODEL

#include <vector>

#include <pytorch_util.h>
#include <encoders/go_zero_encoder.h>
#include <models/model_base.h>

#include <torch/torch.h>

namespace rlgames {

namespace s = std;
namespace t = torch;

struct ZeroModelSmallOptions {
  TensorDim  bc1sz;
  TensorDim  bc2sz;
  TensorDim  bc3sz;
  TensorDim  bc4sz;
  TensorDim  pcsz;
  TensorDim  vcsz;

  ZeroModelSmallOptions() = default;
  ZeroModelSmallOptions(
    TensorDim  bc1sz,
    TensorDim  bc2sz,
    TensorDim  bc3sz,
    TensorDim  bc4sz,
    TensorDim  pcsz,
    TensorDim  vcsz):
    bc1sz(bc1sz),
    bc2sz(bc2sz),
    bc3sz(bc3sz),
    bc4sz(bc4sz),
    pcsz(pcsz),
    vcsz(vcsz)
  {}
  ZeroModelSmallOptions(const s::vector<TensorDim> dims){
    assert(dims.size() == 6);

    bc1sz = dims[0];
    bc2sz = dims[1];
    bc3sz = dims[2];
    bc4sz = dims[3];
    pcsz  = dims[4];
    vcsz  = dims[5];
  }
};

class ZeroModelSmallImpl : public t::nn::Module {
  t::nn::Conv2d bc1, bc2, bc3, bc4, pc1, vc1;
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
    board = t::relu(bc1(board));
    board = t::relu(bc2(board));
    board = t::relu(bc3(board));
    board = t::relu(bc4(board));
    return board;
  }
public:
  ZeroModelSmallImpl(
    TensorDimP ssz,
    TensorDim  bc1sz,
    TensorDim  bc2sz,
    TensorDim  bc3sz,
    TensorDim  bc4sz,
    TensorDim  pcsz,
    TensorDim  vcsz,
    uint asz):
    bc1(register_module("bc1", t::nn::Conv2d(t::nn::Conv2dOptions(ssz.x.i, bc1sz.i, bc1sz.j)))),
    bc2(register_module("bc2", t::nn::Conv2d(t::nn::Conv2dOptions(bc1sz.i, bc2sz.i, bc2sz.j)))),
    bc3(register_module("bc3", t::nn::Conv2d(t::nn::Conv2dOptions(bc2sz.i, bc3sz.i, bc3sz.j)))),
    bc4(register_module("bc4", t::nn::Conv2d(t::nn::Conv2dOptions(bc3sz.i, bc4sz.i, bc4sz.j)))),
    pc1(register_module("pc1", t::nn::Conv2d(t::nn::Conv2dOptions(bc4sz.i, pcsz.i, pcsz.j)))),
    vc1(register_module("vc1", t::nn::Conv2d(t::nn::Conv2dOptions(bc4sz.i, vcsz.j, vcsz.j)))),
    pl1(register_module("pl1",
      t::nn::Linear(
        (ssz.x.j - bc1sz.j - bc2sz.j - bc3sz.j - bc4sz.j - pcsz.j + 5) *
        (ssz.x.k - bc1sz.k - bc2sz.k - bc3sz.k - bc4sz.k - pcsz.k + 5) *
        pcsz.i + ssz.y.i,
      asz))),
    vl1(register_module("vl1",
      t::nn::Linear(
        (ssz.x.j - bc1sz.j - bc2sz.j - bc3sz.j - bc4sz.j - vcsz.j + 5) *
        (ssz.x.k - bc1sz.k - bc2sz.k - bc3sz.k - bc4sz.k - vcsz.k + 5) *
        vcsz.i + ssz.y.i,
      1))),
    state_size(ssz)
  {}
  ZeroModelSmallImpl(TensorDimP ssz, uint asz, const ZeroModelSmallOptions& opt):
    bc1(register_module("bc1", t::nn::Conv2d(t::nn::Conv2dOptions(ssz.x.i, opt.bc1sz.i, opt.bc1sz.j)))),
    bc2(register_module("bc2", t::nn::Conv2d(t::nn::Conv2dOptions(opt.bc1sz.i, opt.bc2sz.i, opt.bc2sz.j)))),
    bc3(register_module("bc3", t::nn::Conv2d(t::nn::Conv2dOptions(opt.bc2sz.i, opt.bc3sz.i, opt.bc3sz.j)))),
    bc4(register_module("bc4", t::nn::Conv2d(t::nn::Conv2dOptions(opt.bc3sz.i, opt.bc4sz.i, opt.bc4sz.j)))),
    pc1(register_module("pc1", t::nn::Conv2d(t::nn::Conv2dOptions(opt.bc4sz.i, opt.pcsz.i, opt.pcsz.j)))),
    vc1(register_module("vc1", t::nn::Conv2d(t::nn::Conv2dOptions(opt.bc4sz.i, opt.vcsz.i, opt.vcsz.j)))),
    pl1(register_module("pl1",
      t::nn::Linear(
        (ssz.x.j - opt.bc1sz.j - opt.bc2sz.j - opt.bc3sz.j - opt.bc4sz.j - opt.pcsz.j + 5) *
        (ssz.x.k - opt.bc1sz.k - opt.bc2sz.k - opt.bc3sz.k - opt.bc4sz.k - opt.pcsz.k + 5) *
        opt.pcsz.i + ssz.y.i,
      asz))),
    vl1(register_module("vl1",
      t::nn::Linear(
        (ssz.x.j - opt.bc1sz.j - opt.bc2sz.j - opt.bc3sz.j - opt.bc4sz.j - opt.vcsz.j + 5) *
        (ssz.x.k - opt.bc1sz.k - opt.bc2sz.k - opt.bc3sz.k - opt.bc4sz.k - opt.vcsz.k + 5) *
        opt.vcsz.i + ssz.y.i,
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
TORCH_MODULE(ZeroModelSmall);

} // rlgames

#endif//RLGAMES_ZERO_MODEL
