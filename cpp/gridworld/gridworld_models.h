#ifndef GRIDWORLD_MODELS
#define GRIDWORLD_MODELS

#include <torch/torch.h>

#include <gridworld.h>

#include <cassert>
#include <ctime>
#include <cstdlib>
#include <memory>

namespace gridworld_pt {

namespace g = gridworld;
namespace t = torch;

enum class GridEnvMode : uint {
  StaticSimple,
  RandomPlayerLocation,
  RandomSimple,
};

class GridEnv {
  uint        mSize;
  GridEnvMode mMode;

  g::GridWorld static_simple_init() const {
    return g::GridWorld(mSize, 1, 1, 1, 0, true);
  }

  g::GridWorld random_player_location_init() const {
    g::GridWorld ret(mSize, 1, 1, 1, 0, true);
    s::srand(time(0));
    while (not ret.set_player_location(g::Pt(s::rand() % mSize, s::rand() % mSize)));
    return ret;
  }

  g::GridWorld random_simple_init() const {
    s::srand(time(0));
    return g::GridWorld(mSize, 1, 1, 1, rand(), true);
  }
public:
  GridEnv(uint sz, GridEnvMode mode): mSize(sz), mMode(mode) {}

  g::GridWorld create() const {
    switch (mMode){
    case GridEnvMode::StaticSimple:
      return static_simple_init();
    case GridEnvMode::RandomPlayerLocation:
      return random_player_location_init();
    case GridEnvMode::RandomSimple:
      return random_simple_init();
    default: assert(false);
    }
  }

  const g::GridState& get_state(const g::GridWorld& ins) const {
    return ins.get_state();
  }

  uint size() const {
    return mSize;
  }

  uint state_size() const {
    return mSize * mSize * 4;
  }

  uint action_size() const {
    return (uint)g::Action::MAX;
  }

  float get_reward(const g::GridWorld& ins) const {
    float reward = ins.get_reward();
    if (reward == 0.)
      return -1.f;
    else
      return reward;
  }

  void apply_action(g::GridWorld& ins, g::Action action) const {
    ins.move(action);
  }

  bool is_termination(g::GridWorld& ins) const {
    return ins.is_complete();
  }

  void display(g::GridWorld& ins) const {
    ins.print(s::cout);
  }
};

class GridStateEncoder {
  const GridEnv& mEnv;
  mutable float* mState;
public:
  explicit GridStateEncoder(const GridEnv& env):
    mEnv(env), mState(nullptr) {
    mState = new float[env.state_size() * 4];
  }
  GridStateEncoder(GridStateEncoder&& o): mEnv(o.mEnv), mState(o.mState) {
    o.mState = nullptr;
  }
  ~GridStateEncoder(){
    if (mState){
      delete[] mState;
      mState = nullptr;
    }
  }

  t::Tensor encode_state(const g::GridState& state) const {
    uint sz = mEnv.size();
    uint psz = sz * sz;
    uint encoded_state_sz = sz * sz * 4;
    //Plane1: player
    //Plane2: walls
    //Plane3: sinks
    //Plane4: goals
    for (uint i = 0; i < mEnv.state_size(); ++i)
      mState[i] = 0.;
    for (uint i = 0; i < sz * sz; ++i)
      switch (state.get(i)){
        case g::Obj::Empty: break;
        case g::Obj::Player:
          mState[i] = 1.f;
          break;
        case g::Obj::Wall:
          mState[i + psz] = 1.f;
          break;
        case g::Obj::Sink:
          mState[i + psz * 2] = 1.f;
          break;
        case g::Obj::Goal:
          mState[i + psz * 3] = 1.f;
          break;
        case g::Obj::PS:
          mState[i] = 1.f;
          mState[i + psz * 2] = 1.f;
          break;
        case g::Obj::PG:
          mState[i] = 1.f;
          mState[i + psz * 3] = 1.f;
          break;
        default: assert(false);
      }
    t::Tensor ret = t::from_blob(mState, {encoded_state_sz});
    return ret;
  }

  g::GridState decode_state(t::Tensor tensor) const {
    uint sz = mEnv.size();
    uint encoded_state_sz = sz * sz * 4;
    s::memcpy(mState, tensor.data_ptr(), encoded_state_sz);
    g::GridState ret(sz);

    //Plane1: player
    //Plane2: walls
    //Plane3: sinks
    //Plane4: goals
    for (uint i = sz * sz; i < sz * sz * 2; ++i)
      if (mState[i] == 1.f)
        ret.set_cell(g::Obj::Wall, i - sz * sz);
    for (uint i = sz * sz * 2; i < sz * sz * 3; ++i)
      if (mState[i] == 1.f)
        ret.set_cell(g::Obj::Sink, i - sz * sz * 2);
    for (uint i = sz * sz * 3; i < sz * sz * 4; ++i)
      if (mState[i] == 1.f)
        ret.set_cell(g::Obj::Goal, i - sz * sz * 3);
    for (uint i = 0; i < sz * sz; ++i)
      if (mState[i] == 1.f)
        ret.set_cell(g::Obj::Player, i);
    return ret;
  }
};

class GridActionEncoder {
  const GridEnv& mEnv;
  mutable float* mAction;
public:
  explicit GridActionEncoder(const GridEnv& env):
    mEnv(env), mAction(nullptr) {
    mAction = new float[env.action_size()];
  }
  GridActionEncoder(GridActionEncoder&& o): mEnv(o.mEnv), mAction(o.mAction) {
    o.mAction = nullptr;
  }
  ~GridActionEncoder(){
    if (mAction){
      delete[] mAction;
      mAction = nullptr;
    }
  }

  t::Tensor encode_action(g::Action action) const {
    uint sz = mEnv.action_size();
    for (uint i = 0; i < sz; ++i)
      mAction[i] = 0.f;

    switch (action){
    case g::Action::UP:
      mAction[0] = 1.f;
      break;
    case g::Action::DN:
      mAction[1] = 1.f;
      break;
    case g::Action::LF:
      mAction[2] = 1.f;
      break;
    case g::Action::RT:
      mAction[3] = 1.f;
      break;
    default: assert(false);
    }
    t::Tensor ret = t::from_blob(mAction, {sz});
    return ret;
  }

  g::Action decode_action(t::Tensor tensor) const {
    uint max_idx = t::argmax(tensor).item().to<int>();

    assert(max_idx < mEnv.action_size());

    switch (max_idx){
    case 0: return g::Action::UP;
    case 1: return g::Action::DN;
    case 2: return g::Action::LF;
    case 3: return g::Action::RT;
    default: assert(false);
    }
  }
};

class SimpleGridModelImpl : public t::nn::Module {
  t::nn::Linear l1, l2, l3;
  t::Tensor     b1, b2;
public:
  SimpleGridModelImpl(sint64 isz, sint64 l1sz, sint64 l2sz, sint64 osz):
    l1(register_module("l1", t::nn::Linear(isz, l1sz))),
    l2(register_module("l2", t::nn::Linear(l1sz, l2sz))),
    l3(register_module("l3", t::nn::Linear(l2sz, osz))),
    b1(register_parameter("b1", t::randn(l1sz))),
    b2(register_parameter("b2", t::randn(l2sz)))
  {}

  t::Tensor forward(t::Tensor x){
    x = t::relu(l1(x) + b1);
    x = t::relu(l2(x) + b2);
    x = l3(x);
    return x;
  }
};
TORCH_MODULE(SimpleGridModel);

class MediumGridModelImpl : public t::nn::Module {
  t::nn::Linear l1, l2, l3, l4;
  t::Tensor     b1, b2, b3;
public:
  MediumGridModelImpl(sint64 isz, sint64 l1sz, sint64 l2sz, sint64 l3sz, sint64 osz):
    l1(register_module("l1", t::nn::Linear(isz, l1sz))),
    l2(register_module("l2", t::nn::Linear(l1sz, l2sz))),
    l3(register_module("l3", t::nn::Linear(l2sz, l3sz))),
    l4(register_module("l4", t::nn::Linear(l3sz, osz))),
    b1(register_parameter("b1", t::randn(l1sz))),
    b2(register_parameter("b2", t::randn(l2sz))),
    b3(register_parameter("b3", t::randn(l3sz)))
  {}

  t::Tensor forwar(t::Tensor x){
    x = t::relu(l1(x) + b1);
    x = t::relu(l2(x) + b2);
    x = t::relu(l3(x) + b3);
    x = l4(x);
    return x;
  }
};
TORCH_MODULE(MediumGridModel);

template <typename NNModel, typename SE, typename AE, typename Optim>
struct RLModel {
  NNModel model;
  SE      state_encoder;
  AE      action_encoder;
  Optim   optimizer;
public:
  RLModel(NNModel m, SE&& se, AE&& ae, float learning_rate):
    model(m),
    state_encoder(s::move(se)),
    action_encoder(s::move(ae)),
    optimizer(m->parameters(), t::optim::AdamOptions(learning_rate))
  {}
};

} // gridworld_pt

#endif//GRIDWORLD_MODELS
