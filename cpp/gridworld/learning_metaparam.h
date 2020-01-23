#ifndef GRIDWORLD_LEARNING_PARAM
#define GRIDWORLD_LEARNING_PARAM

#include <type_alias.h>

namespace gridworld_pt {

struct epsilon_greedy_metaparams {
  double epsilon;
};

struct experience_replay_metaparams {
  uint sz;
  uint batchsize;
};

struct prioritized_experience_replay_metaparams {
  uint sz;
  uint batchsize;
  float alpha;
  float init_delta;
};

template <typename EParams, typename ERBParams>
struct qlearning_metaparams {
  uint64    epochs;
  double    gamma;
  uint      max_steps;
  uint      tc_steps;
  EParams   exp;
  ERBParams erb;
};

template <typename EParams, typename ERBParams>
struct distqlearning_metaparams {
  uint64    epochs;
  double    gamma;
  uint      max_steps;
  uint      tc_steps;
  uint      reward_dist_slices;
  EParams   exp;
  ERBParams erb;
};

struct pg_metaparams {
  uint64 epochs;
  uint64 batchsize;
  double gamma;
  uint   max_steps;
};

struct ac_metaparams {
  uint64 epochs;
  uint64 batchsize;
  double gamma;
  uint   max_steps;
  uint   tc_steps;
};

} // gridworld_pt

#endif//GRIDWORLD_LEARNING_PARAM
