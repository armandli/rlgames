#ifndef GRIDWORLD_EXPERIENCE3
#define GRIDWORLD_EXPERIENCE3

#include <torch/torch.h>

#include <vector>

#include <type_alias.h>

#include <learning_util.h>
#include <experience_util.h>

namespace gridworld_pt {

//experience replay buffer, implemented for state being a convolutional tensor

namespace s = std;
namespace t = torch;

template <typename ACTION>
class ExpReplayBuffer3 {
  t::Tensor                  mStateBuf;
  t::Tensor                  mNStateBuf;
  s::vector<ARTuple<ACTION>> mTuples;
  Dim                        mStateSize;
  uint64                     mNextIdx;
  t::Device                  mDevice;
  bool                       mFilled;

  void random_permutation(s::vector<uint64>& idxes, uint sz){
    for (size_t i = 0; i < s::min(sz, (uint)idxes.size() - 1); ++i){
      size_t ridx = rand() % (idxes.size() - 1 - i) + 1 + i;
      idxes[i]    ^= idxes[ridx];
      idxes[ridx] ^= idxes[i];
      idxes[i]    ^= idxes[ridx];
    }
    idxes.erase(s::begin(idxes) + sz, s::end(idxes));
  }

  s::vector<ARTuple<ACTION>> index_select(const s::vector<ARTuple<ACTION>>& v, const s::vector<uint64>& indicies){
    s::vector<ARTuple<ACTION>> ret(indicies.size());
    for (uint i = 0; i < indicies.size(); ++i)
      ret[i] = v[indicies[i]];
    return ret;
  }

  ExpReplayBuffer3(t::Tensor states, t::Tensor nstates, const s::vector<ARTuple<ACTION>> tuples, Dim state_size, t::Device device):
    mStateBuf(states),
    mNStateBuf(nstates),
    mTuples(std::move(tuples)),
    mStateSize(state_size),
    mNextIdx(0),
    mDevice(device),
    mFilled(true)
  {}
public:
  ExpReplayBuffer3(uint64 sz, Dim state_size, t::Device device):
    mStateBuf(t::zeros({(sint64)sz, (sint64)state_size.x, (sint64)state_size.y, (sint64)state_size.z}, device)),
    mNStateBuf(t::zeros({(sint64)sz, (sint64)state_size.x, (sint64)state_size.y, (sint64)state_size.z}, device)),
    mTuples(sz),
    mStateSize(state_size),
    mNextIdx(0),
    mDevice(device),
    mFilled(false)
  {}
  void append(const Exp<ACTION>& exp){
    mStateBuf[mNextIdx] = exp.tstate;
    mNStateBuf[mNextIdx] = exp.ntstate;
    mTuples[mNextIdx] = ARTuple<ACTION>(exp.action, exp.reward, exp.ntstate_isterminal);
    mNextIdx++;
    if (mNextIdx >= mTuples.size()){
      mNextIdx = 0;
      mFilled = true;
    }
  }
  bool is_filled() const {
    return mFilled;
  }
  ExpReplayBuffer3 sample_batch(uint64 batchsize){
    assert(batchsize <= mTuples.size());

    s::vector<uint64> indicies(mTuples.size());
    s::iota(s::begin(indicies), s::end(indicies), 0);
    random_permutation(indicies, batchsize);

    s::vector<ARTuple<ACTION>> tuples = index_select(mTuples, indicies);

    t::Tensor index = t::from_blob(indicies.data(), {(sint64)indicies.size()}, t::kLong);
    t::Tensor index_dev = index.to(mDevice);
    index_dev = index_dev.repeat_interleave(mStateSize.flatten_size());
    index_dev = index_dev.reshape({(sint64)batchsize, mStateSize.x, mStateSize.y, mStateSize.z});

    t::Tensor batch_states = t::gather(mStateBuf, 0, index_dev);
    t::Tensor batch_nstates = t::gather(mNStateBuf, 0, index_dev);

    return ExpReplayBuffer3(batch_states, batch_nstates, tuples, mStateSize, mDevice);
  }

  t::Tensor states_tensor() { return mStateBuf; }
  t::Tensor nstates_tensor() { return mNStateBuf; }
  ARTArray actions_n_rewards() const {
    ARTArray ret(mTuples.size());
    for (uint i = 0; i < mTuples.size(); ++i){
      ret.actions[i] = (long)mTuples[i].action;
      ret.rewards[i] = mTuples[i].reward;
      ret.is_terminals[i] = mTuples[i].is_terminal;
    }
    return ret;
  }
};

} //gridworld_pt

#endif//GRIDWORLD_EXPERIENCE3
