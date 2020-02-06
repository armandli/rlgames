#ifndef GRIDWORLD_EXPERIENCE_BUFFER2
#define GRIDWORLD_EXPERIENCE_BUFFER2

#include <torch/torch.h>

#include <type_alias.h>

#include <learning_util.h>
#include <experience_util.h>

namespace gridworld_pt {

// experience buffer for convolutional states

namespace s = std;
namespace t = torch;

template <typename ACTION>
class ExpBuffer2 {
  t::Tensor                  mStateBuf;
  t::Tensor                  mNStateBuf;
  s::vector<ARTuple<ACTION>> mTuples;
  uint64                     mNextIdx;
public:
  ExpBuffer2(uint64 sz, Dim state_size, t::Device device):
    mStateBuf(t::zeros({(sint64)sz, (sint64)state_size.x, (sint64)state_size.y, (sint64)state_size.z}, device)),
    mNStateBuf(t::zeros({(sint64)sz, (sint64)state_size.x, (sint64)state_size.y, (sint64)state_size.z}, device)),
    mTuples(sz),
    mNextIdx(0)
  {}
  bool append(const Exp<ACTION>& exp){
    if (mNextIdx >= mTuples.size()) return false;

    mStateBuf[mNextIdx] = exp.tstate;
    mNStateBuf[mNextIdx] = exp.ntstate;
    mTuples[mNextIdx] = ARTuple<ACTION>(exp.action, exp.reward, exp.ntstate_isterminal);
    mNextIdx++;
    return true;
  }
  void reset(){
    mNextIdx = 0;
  }

  t::Tensor states_tensor(){ return mStateBuf.slice(0, 0, mNextIdx); }
  t::Tensor nstates_tensor(){ return mNStateBuf.slice(0, 0, mNextIdx); }
  ARTArray actions_n_rewards() const {
    ARTArray ret(mNextIdx);
    for (uint i = 0; i < mNextIdx; ++i){
      ret.actions[i] = (long)mTuples[i].action;
      ret.rewards[i] = mTuples[i].reward;
      ret.is_terminals[i] = mTuples[i].is_terminal;
    }
    return ret;
  }
};

} // gridworld_pt

#endif//GRIDWORLD_EXPERIENCE_BUFFER2
