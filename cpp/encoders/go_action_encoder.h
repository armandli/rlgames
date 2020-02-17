#ifndef RLGAMES_GO_ACTION_ENCODER
#define RLGAMES_GO_ACTION_ENCODER

#include <type_alias.h>
#include <types.h>
#include <encoders/encoder_base.h>

#include <torch/torch.h>

#include <cassert>
#include <algorithm>

namespace rlgames {

namespace s = std;
namespace t = torch;

template <ubyte SZ>
class ZeroGoActionEncoder : public ActionEncoderBase<Move, ZeroGoActionEncoder<SZ>> {
  mutable float mAction[SZ * SZ + 1];
public:
  t::Tensor encode_action(const Move& a, t::Device device) const {
    uint sz = action_size();
    s::fill(mAction, mAction + SZ * SZ + 1, 0.F);

    if (a.mty == M::Play)
      mAction[index<SZ>(a.mpt)] = 1.f;
    else
      mAction[SZ * SZ] = 1.f;
    t::Tensor m = t::from_blob(mAction, {sz});
    if (device.type() == t::kCUDA)
      return m.to(device);
    else
      return m.clone();
  }
  Move decode_action(t::Tensor tensor) const {
    uint maxidx = t::argmax(tensor).item().to<int>();
    assert(maxidx < action_size());
    if (maxidx == SZ * SZ) return Move(M::Pass);
    else                   return Move(M::Play, point<SZ>(maxidx));
  }
  uint move_to_idx(const Move& m) const {
    if (m.mty == M::Pass || m.mty == M::Resign) return SZ * SZ;
    else                                        return index<SZ>(m.mpt);
  }
  Move idx_to_move(uint idx) const {
    assert(idx <= SZ * SZ);

    if (idx >= SZ * SZ) return Move(M::Pass);
    else                return Move(M::Play, point<SZ>(idx));
  }
  static constexpr uint action_size(){
    return SZ * SZ + 1;
  }
};

} //rlgames

#endif//RLGAMES_GO_ACTION_ENCODER
