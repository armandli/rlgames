#ifndef RLGAMES_ZERO_EPISODIC_BUFFER
#define RLGAMES_ZERO_EPISODIC_BUFFER

#include <cassert>
#include <string>
#include <vector>
#include <algorithm>

#include <type_alias.h>
#include <pytorch_util.h>
#include <encoders/go_zero_encoder.h>

#include <torch/torch.h>

namespace rlgames {

namespace s = std;
namespace t = torch;

struct ZeroExperience {
  t::Tensor boards;
  t::Tensor states;
  t::Tensor visit_counts;
  t::Tensor rewards;

  ZeroExperience(t::Tensor b, t::Tensor s, t::Tensor vc, t::Tensor r):
    boards(b), states(s), visit_counts(vc), rewards(r) {}
  ZeroExperience(): boards(t::zeros({0})), states(t::zeros({0})), visit_counts(t::zeros({0})), rewards(t::zeros({0})) {}

  void export_experience(const s::string& boards_file, const s::string& states_file, s::string& vcount_file, s::string& rewards_file){
    t::save(boards, boards_file);
    t::save(states, states_file);
    t::save(visit_counts, vcount_file);
    t::save(rewards, rewards_file);
  }
};

class ZeroEpisodicExpCollector {
  t::Tensor        mBoards;
  t::Tensor        mStates;
  s::vector<float> mVisitCounts;
  s::vector<float> mRewards;
  t::Device        mDevice;
  uint             mOffset;
  uint             mStepCount;
  uint             mActionSize;
public:
  ZeroEpisodicExpCollector(uint max_size, const TensorDimP& state_size, uint action_size, t::Device device):
    mBoards(t::zeros({max_size, state_size.x.i, state_size.x.j, state_size.x.k}, device)),
    mStates(t::zeros({max_size, state_size.y.i}, device)),
    mRewards(max_size),
    mDevice(device),
    mOffset(0),
    mStepCount(0),
    mActionSize(action_size){
    mVisitCounts.reserve(max_size * action_size);
  }
  bool append(TensorP st, const s::vector<float>& visit_counts){
    if (mOffset + mStepCount >= mRewards.size())
      return false;

    mBoards[mOffset + mStepCount] = st.x;
    mStates[mOffset + mStepCount] = st.y;
    s::copy(s::begin(visit_counts), s::end(visit_counts), s::back_inserter(mVisitCounts));

    mStepCount++;
    return true;
  }
  void complete_episode(float reward){
    s::fill(s::begin(mRewards) + mOffset, s::begin(mRewards) + mOffset + mStepCount, reward);

    mOffset += mStepCount;
    mStepCount = 0;
  }

  t::Tensor boards(){
    return mBoards.slice(0, 0, mOffset);
  }
  t::Tensor states(){
    return mStates.slice(0, 0, mOffset);
  }
  t::Tensor visit_counts(){
    t::Tensor vc = t::from_blob(mVisitCounts.data(), {(sint64)mOffset, (sint64)mActionSize});
    if (mDevice == t::kCPU)
      return vc.clone();
    else
      return vc.to(mDevice);
  }
  t::Tensor rewards(){
    t::Tensor r = t::from_blob(mRewards.data(), {(sint64)mOffset});
    if (mDevice == t::kCPU)
      return r.clone();
    else
      return r.to(mDevice);
  }
  ZeroExperience experience(){
    assert(mVisitCounts.size() == mOffset * mActionSize);

    return ZeroExperience(boards(), states(), visit_counts(), rewards());
  }
};

void append_experiences(ZeroExperience& out, ZeroEpisodicExpCollector& collector){
  ZeroExperience nexp = collector.experience();
  out.boards = t::cat({out.boards, nexp.boards}, 0);
  out.states = t::cat({out.states, nexp.states}, 0);
  out.visit_counts = t::cat({out.visit_counts, nexp.visit_counts}, 0);
  out.rewards = t::cat({out.rewards, nexp.rewards}, 0);
}

void append_experiences(ZeroExperience& out, ZeroEpisodicExpCollector& c1, ZeroEpisodicExpCollector& c2){
  ZeroExperience nexp1 = c1.experience();
  ZeroExperience nexp2 = c2.experience();
  out.boards = t::cat({out.boards, nexp1.boards, nexp2.boards}, 0);
  out.states = t::cat({out.states, nexp1.states, nexp2.states}, 0);
  out.visit_counts = t::cat({out.visit_counts, nexp1.visit_counts, nexp2.visit_counts}, 0);
  out.rewards = t::cat({out.rewards, nexp1.rewards, nexp2.rewards}, 0);
}

} // rlgames

#endif//RLGAMES_ZERO_EPISODIC_BUFFER
