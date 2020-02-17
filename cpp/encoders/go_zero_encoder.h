#ifndef RLGAMES_GO_ZERO_ENCODER
#define RLGAMES_GO_ZERO_ENCODER

#include <type_alias.h>
#include <types.h>
#include <game_base.h>
#include <go_types.h>
#include <pytorch_util.h>
#include <encoders/encoder_base.h>

#include <torch/torch.h>

#include <cassert>
#include <algorithm>

namespace rlgames {

namespace s = std;
namespace t = torch;

// AlphaZero Go State Encoder
// Plane 0-3 are our stones with 1,2,3,4+ liberties
// Plane 4-7 are their stones with 1,2,3,4+ liberties
// Plane 8 are illegal moves due to ko
// Additional 2 dim out-of-plane information on if player get komi (1 if yes) and if opponent gets komi,
// maybe this is better with the actual komi value

template <ubyte SZ>
class ZeroGoStateEncoder : public StateEncoderBase<GoGameState<SZ>, TensorP, ZeroGoStateEncoder<SZ>> {
  mutable float mBoard[SZ * SZ * 9];
  mutable float mState[2];

  static constexpr uint IZ = SZ * SZ;
public:
  TensorP encode_state(const GoGameState<SZ>& gs, t::Device device){
    const GoBoard<SZ>& board = gs.board();
    Player nplayer = gs.next_player();
    s::fill(mBoard, mBoard + SZ * SZ * 9, 0.f);
    for (uint i = 0; i < SZ; ++i)
      for (uint j = 0; j < SZ; ++j){
        Pt pt(i, j);
        uint idx = index<SZ>(pt);
        const GoStr<SZ>* string = board.get_string(pt);
        Player player = board.get(pt);
        if (string){
          switch (string->num_liberties()){
          case 0U: assert(false); //should not exist
          case 1U:
            if (player == nplayer) mBoard[idx]          = 1.f;
            else                   mBoard[idx + IZ * 4] = 1.f;
            break;
          case 2U:
            if (player == nplayer) mBoard[idx + IZ]     = 1.f;
            else                   mBoard[idx + IZ * 5] = 1.f;
            break;
          case 3U:
            if (player == nplayer) mBoard[idx + IZ * 2] = 1.f;
            else                   mBoard[idx + IZ * 6] = 1.f;
            break;
          default:
            if (player == nplayer) mBoard[idx + IZ * 3] = 1.f;
            else                   mBoard[idx + IZ * 7] = 1.f;
          }
        } else if (gs.does_move_violate_ko(Move(M::Play, pt))){
            mBoard[idx + IZ * 8] = 1.f;
        }
      }
    // use default komi instead of 1.0F
    if (nplayer == Player::White){
      mState[0] = default_komi<SZ>();
      mState[1] = 0.f;
    } else {
      mState[0] = 0.f;
      mState[1] = default_komi<SZ>();
    }

    t::Tensor tboard = t::from_blob(mBoard, {9, (sint)SZ, (sint)SZ});
    t::Tensor tstate = t::from_blob(mState, {2});
    if (device.type() == t::kCUDA)
      return TensorP(tboard.to(device), tstate.to(device));
    else
      return TensorP(tboard.clone(), tstate.clone());
  }
  TensorDimP state_size() const {
    return TensorDimP(TensorDim(9, SZ, SZ), TensorDim(2));
  }
};

} //rlgames

#endif//RLGAMES_GO_ZERO_ENCODER
