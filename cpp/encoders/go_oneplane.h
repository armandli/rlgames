#ifndef RLGAMES_GO_ONEPLANE
#define RLGAMES_GO_ONEPLANE

#include <type_alias.h>
#include <encoder_base.h>
#include <types.h>
#include <game_base.h>
#include <go_types.h>

#include <torch/torch.h>

#include <cassert>

namespace rlgames {

namespace t = torch;

//encode the board as SZ-by-SZ dimensional input plane
template <ubyte SZ>
class OnePlaneGoStateEncoder : public StateEncoderBase<GoGameState<SZ>, t::Tensor, OnePlaneGoStateEncoder<SZ>> {
  mutable float mState[SZ * SZ];

  constexpr uint plane_size() const {
    return SZ * SZ;
  }
public:
  t::Tensor encode_state(const GoGameState<SZ>& gs, t::Device device) const {
    const GoBoard<SZ>& board = gs.board();
    for (uint i = 0; i < SZ; ++i)
      for (uint j = 0; j < SZ; ++j){
        Pt pt(i, j);
        switch (board.get(pt)){
        case Player::Black:
        case Player::White:
          if (board.get(Pt(i, j)) == gs.next_player())
            mState[index<SZ>(pt)] = 1.f;
          else
            mState[index<SZ>(pt)] = -1.f;
          break;
        case Player::Unknown:
          mState[index<SZ>(pt)] = 0.f;
          break;
        default: assert(false);
        }
      }
    t::Tensor m = t::from_blob(mState, {(sint)SZ, (sint)SZ});
    if (device.type() == t::kCUDA)
      return m.to(device);
    else
      return m.clone();
  }
  TensorDimP state_size() const {
    return TensorDimP(TensorDim(SZ, SZ), TensorDim(0));
  }
};

} //rlgames

#endif//RLGAMES_GO_ONEPLANE
