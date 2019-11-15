#ifndef GRIDWORLD_EXPERIENCE
#define GRIDWORLD_EXPERIENCE

#include <torch/torch.h>

#include <cassert>
#include <random>
#include <vector>
#include <algorithm>

#include <type_alias.h>

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

template <typename ACTION>
struct Exp {
  t::Tensor tstate;
  t::Tensor ntstate;
  float     reward;
  ACTION    action;

  Exp(): tstate(), ntstate(), reward(0.f), action((ACTION)0) {}
  Exp(t::Tensor st, ACTION a, float r, t::Tensor nst):
    tstate(st.clone()), ntstate(nst.clone()), reward(r), action(a)
  {}
};

template <typename Exp> class ExpReplayBuffer;

template <typename Exp>
class ExpBatch {
  ExpReplayBuffer<Exp>& mERB;
  s::vector<uint64>     mIdxes;

  class Iterator {
    ExpBatch& mBatch;
    uint64 mPos;
  public:
    explicit Iterator(ExpBatch& batch, uint64 position): mBatch(batch), mPos(position) {}

    Exp& operator*(){
      return mBatch.mERB[mBatch.mIdxes[mPos]];
    }
    Iterator& operator++(){
      mPos = s::min(mPos + 1, mBatch.mIdxes.size());
      return *this;
    }
    bool operator !=(const Iterator& it){
      return mPos != it.mPos;
    }
    //TODO: implement index operator []
  };
public:
  ExpBatch(ExpReplayBuffer<Exp>& b, const s::vector<uint64>& idxes): mERB(b), mIdxes(idxes) {}

  Iterator begin(){
    return Iterator(*this, 0);
  }
  Iterator end(){
    return Iterator(*this, mIdxes.size());
  }
};

template <typename Exp>
class ExpReplayBuffer {
  s::vector<Exp> mBuffer;
  uint64         mNextIdx;
  bool           mFilled;

  void random_permutation(s::vector<uint64>& idxes, uint sz){
    for (size_t i = 0; i < s::min(sz, (uint)idxes.size() - 1); ++i){
      size_t ridx = rand() % (idxes.size() - 1 - i) + 1 + i;
      idxes[i]    ^= idxes[ridx];
      idxes[ridx] ^= idxes[i];
      idxes[i]    ^= idxes[ridx];
    }
    idxes.erase(s::begin(idxes) + sz, s::end(idxes));
  }
public:
  explicit ExpReplayBuffer(uint64 sz): mBuffer(sz), mNextIdx(0), mFilled(false) {}
  void append(const Exp& exp){
    mBuffer[mNextIdx++] = exp;
    if (mNextIdx >= mBuffer.size()){
      mNextIdx = 0;
      mFilled = true;
    }
  }
  bool is_filled() const {
    return mFilled;
  }
  Exp& operator[](uint idx){
    assert(idx < mBuffer.size());
    return mBuffer[idx];
  }
  ExpBatch<Exp> sample_batch(uint64 batchsize){
    assert(batchsize <= mBuffer.size());
    //TODO: can we make this better with no memory usage?
    s::vector<uint64> indicies(mBuffer.size());
    s::iota(s::begin(indicies), s::end(indicies), 0);
    random_permutation(indicies, batchsize);
    //TODO: move the vector, don't copy
    return ExpBatch<Exp>(*this, indicies);
  }
};

} // gridworld_pt

#endif//GRIDWORLD_EXPERIENCE
