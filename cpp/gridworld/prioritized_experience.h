#ifndef GRIDWORLD_PRIORITIZED_EXPERIENCE
#define GRIDWORLD_PRIORITIZED_EXPERIENCE

#include <torch/torch.h>

#include <cmath>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>

#include <type_alias.h>

#include <experience_util.h>

//implements prioritized experience replay buffer using Bellman error; does not add bias correction beta in this version

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

class SumTree {
  s::vector<float> mSums;
  s::vector<float> mLeafs;
  uint             mModPadding;
  float            mMax;

  static uint nearest_power(uint size){
    uint v = size - 1;
    v |= v >> 1U;
    v |= v >> 2U;
    v |= v >> 4U;
    v |= v >> 8U;
    v |= v >> 16U;
    v++;
    return v;
  }

  size_t leaf_size() const {
    return mLeafs.size();
  }

  void init_tree(float init_val){
    mSums.resize(nearest_power(leaf_size()) - 1); //upper bound size
    class Recursion {
      s::vector<float>& sums;
      float             init_val;
    public:
      Recursion(s::vector<float>& sums, float init_val): sums(sums), init_val(init_val) {}
      uint operator()(uint size, uint pos){
        uint first_half, second_half;
        switch (size){
          case 0: case 1: return 0U;
          case 2:
            first_half = second_half = 1U;
            break;
          default: {
            uint half = nearest_power(size) >> 1U;
            uint quarter = half >> 1U;
            first_half = s::min(quarter, size - half) + quarter;
            second_half = size - first_half;
            break;
          }
        }

        sums[pos] = (float)first_half * init_val;

        uint s1 = (*this)(first_half, pos * 2 + 1);
        uint s2 = (*this)(second_half, pos * 2 + 2);
        uint sz = s::max(s1, s2);
        return s::max(pos + 1, sz);
      }
    } recursion(mSums, init_val);
    uint true_size = recursion(leaf_size(), 0);
    mSums.resize(true_size);
  }

  void init_mod_padding(){
    uint idx = 0;
    while (idx < mSums.size())
      idx = idx * 2 + 1;
    mModPadding = leaf_size() - idx % leaf_size();
  }

  uint leaf_idx_to_array_idx(uint leaf_idx) const {
    return (leaf_idx + mModPadding) % leaf_size();
  }
  uint array_idx_to_leaf_idx(uint array_idx) const {
    uint padding = (array_idx + 1 < mModPadding) ? 0U : leaf_size();
    return nearest_power(leaf_size()) - 1 + array_idx - padding;
  }
public:
  SumTree(uint size, float init_val):
    mLeafs(size, init_val), mModPadding(0U), mMax((float)size * init_val){
    assert(init_val > 0.F);
    init_tree(init_val);
    init_mod_padding();
  }
  void update(uint array_idx, float nval){
    assert(array_idx < leaf_size());

    uint leaf_idx = array_idx_to_leaf_idx(array_idx);
    float diff = nval - mLeafs[array_idx];
    mLeafs[array_idx] = nval;
    do {
      if (leaf_idx & 1){
        leaf_idx = (leaf_idx - 1) >> 1U;
        mSums[leaf_idx] += diff;
      } else {
        leaf_idx = (leaf_idx - 2) >> 1U;
      }
    } while (leaf_idx > 0U);
    mMax += diff;
  }
  uint sample(float value) const {
    assert(value < mMax);

    uint idx = 0;
    while (idx < mSums.size())
      if (value < mSums[idx])
        idx = idx * 2 + 1;
      else {
        value -= mSums[idx];
        idx = idx * 2 + 2;
      }

    return leaf_idx_to_array_idx(idx);
  }
  float max() const { return mMax; }
  uint size() const { return leaf_size(); }
};

template <typename ACTION>
class PriExpReplayBatch {
  t::Tensor                  mStateBuf;
  t::Tensor                  mNStateBuf;
  s::vector<ARTuple<ACTION>> mTuples;
  s::vector<uint64>          mIndicies;
public:
  PriExpReplayBatch(t::Tensor states, t::Tensor nstates, const s::vector<ARTuple<ACTION>>& tuples, const s::vector<uint64>& indicies):
    mStateBuf(states),
    mNStateBuf(nstates),
    mTuples(s::move(tuples)),
    mIndicies(indicies)
  {}

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
  const s::vector<uint64>& indicies() const {
    return mIndicies;
  }
};

template <typename ACTION>
class PriExpReplayBuffer {
  t::Tensor                  mStateBuf;
  t::Tensor                  mNStateBuf;
  s::vector<ARTuple<ACTION>> mTuples;
  SumTree                    mTree;
  t::Device                  mDevice;
  s::default_random_engine   mEng;
  uint                       mNextIdx;
  float                      mInitVal;
  float                      mAlpha;
  bool                       mFilled;

  s::vector<ARTuple<ACTION>> index_select(const s::vector<ARTuple<ACTION>>& v, const s::vector<uint64>& indicies){
    s::vector<ARTuple<ACTION>> ret(indicies.size());
    for (uint i = 0; i < indicies.size(); ++i)
      ret[i] = v[indicies[i]];
    return ret;
  }
public:
  PriExpReplayBuffer(uint sz, uint state_size, t::Device device, uint seed, float alpha = 1.F, float init_val = 1.F):
    mStateBuf(t::zeros({(sint64)sz, (sint64)state_size}, device)),
    mNStateBuf(t::zeros({(sint64)sz, (sint64)state_size}, device)),
    mTuples(sz),
    mTree(sz, init_val),
    mDevice(device),
    mEng(seed),
    mNextIdx(0),
    mInitVal(init_val),
    mAlpha(alpha),
    mFilled(false)
  {}
  void append(const Exp<ACTION>& exp){
    mStateBuf[mNextIdx] = exp.tstate;
    mNStateBuf[mNextIdx] = exp.ntstate;
    mTuples[mNextIdx] = ARTuple<ACTION>(exp.action, exp.reward, exp.ntstate_isterminal);
    mTree.update(mNextIdx, s::pow(mInitVal, mAlpha));
    mNextIdx++;
    if (mNextIdx >= mTuples.size()){
      mNextIdx = 0;
      mFilled = true;
    }
  }
  bool is_filled() const {
    return mFilled;
  }
  PriExpReplayBatch<ACTION> sample_batch(uint batchsize){
    assert(batchsize <= mTuples.size());

    s::uniform_real_distribution<float> dist(0.F, mTree.max());
    s::vector<uint64> indicies(batchsize);
    for (uint i = 0; i < batchsize; ++i)
      indicies[i] = mTree.sample(dist(mEng));

    t::Tensor index = t::from_blob(indicies.data(), {(sint64)indicies.size()}, t::kLong);
    t::Tensor index_dev = index.to(mDevice);
    t::Tensor batch_states = t::index_select(mStateBuf, 0, index_dev);
    t::Tensor batch_nstates = t::index_select(mNStateBuf, 0, index_dev);
    s::vector<ARTuple<ACTION>> tuples = index_select(mTuples, indicies);
    return PriExpReplayBatch<ACTION>(batch_states, batch_nstates, tuples, indicies);
  }
  void update(const PriExpReplayBatch<ACTION>& batch, t::Tensor delta_dev){
    t::Device cpu_device(t::kCPU);
    t::Tensor delta = delta_dev.to(cpu_device);
    const s::vector<uint64>& indicies = batch.indicies();
    for (uint i = 0; i < indicies.size(); ++i){
      mTree.update(indicies[i], s::pow(delta[i].item().to<float>(), mAlpha));
    }
  }
};

} // gridworld_pt

#endif//GRIDWORLD_PRIORITIZED_EXPERIENCE
