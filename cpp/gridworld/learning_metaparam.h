#ifndef GRIDWORLD_LEARNING_PARAM
#define GRIDWORLD_LEARNING_PARAM

#include <type_alias.h>

namespace gridworld_pt {

struct epsilon_greedy_metaparams {
  double epsilon;
};

struct experience_replay_metaparams {
  uint64 sz;
  uint   batchsize;
};

template <typename EParams, typename ERBParams>
struct qlearning_metaparams {
  uint64 epochs;
  double gamma;
  uint64 max_steps;
  uint64 tc_steps;
  EParams   exp;
  ERBParams erb;
};

struct pg_metaparams {
  uint64 epochs;
  double gamma;
  uint64 max_steps;
};

} // gridworld_pt

#endif//GRIDWORLD_LEARNING_PARAM
