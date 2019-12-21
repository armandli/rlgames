#ifndef GRIDWORLD_EXPERIENCE_UTIL
#define GRIDWORLD_EXPERIENCE_UTIL

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

template <typename ACTION>
struct Exp {
  t::Tensor tstate;
  t::Tensor ntstate;
  float     reward;
  ACTION    action;
  bool      ntstate_isterminal;

  Exp():
    tstate(), ntstate(), reward(0.f), action((ACTION)0), ntstate_isterminal(false) 
  {}
  Exp(t::Tensor st, ACTION a, float r, t::Tensor nst, bool isterm):
    tstate(st), ntstate(nst), reward(r), action(a), ntstate_isterminal(isterm)
  {}
};

template <typename ACTION>
struct ARTuple {
  ACTION action;
  float  reward;
  bool   is_terminal;

  ARTuple(): action((ACTION)0), reward(0.F), is_terminal(false) {}
  ARTuple(ACTION a, float r, bool t): action(a), reward(r), is_terminal(t) {}
};

struct ARTArray {
  s::vector<long> actions;
  s::vector<float>  rewards;
  s::vector<ubyte>   is_terminals;

  ARTArray() = default;
  explicit ARTArray(uint sz): actions(sz), rewards(sz), is_terminals(sz) {}
};

} // gridworld_pt

#endif//GRIDWORLD_EXPERIENCE_UTIL
