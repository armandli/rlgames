#ifndef GRIDWORLD_MODELS
#define GRIDWORLD_MODELS

#include <torch/torch.h>

#include <gridworld.h>
#include <learning_util.h>

#include <learning_debug.h>

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cstdlib>
#include <memory>
#include <unordered_set>
#include <algorithm>

namespace gridworld_pt {

namespace s = std;
namespace g = gridworld;
namespace t = torch;

enum class GridEnvMode : uint {
  StaticSimple,
  RandomPlayerLocation,
  RandomSimple,
  RandomComplex,
  RandomRepeatedComplex, //repeat the same random generated map a k times
  RandomMaze,
};

class GridEnv {
  uint                           mSize;
  GridEnvMode                    mMode;
  mutable s::unordered_set<uint> mHistory;
  // used for random repeated games, repeating the same random map k times
  uint                           mRepeatTotalCount;
  mutable uint                   mPrevSeed;
  mutable uint                   mRepeatCount;
  //taking each step without reaching the goal have -1 value
  bool                           mUseStepDiscount;
  //we save board history, and give very negative reward for going back to historical position
  bool                           mUseHistoricalMoveDiscount;
  //we terminate the game and return minimal reward if previous state is reached
  bool                           mTerminateOnHistoricalMove;

  g::GridWorld static_simple_init() const {
    return g::GridWorld(mSize, 1, 1, 1, 0, true);
  }

  g::GridWorld random_player_location_init() const {
    g::GridWorld ret(mSize, 1, 1, 1, 0, true);
    while (not ret.set_player_location(g::Pt(s::rand() % mSize, s::rand() % mSize)));
    return ret;
  }

  g::GridWorld random_simple_init() const {
    return g::GridWorld(mSize, 1, 1, 1, rand(), true);
  }

  g::GridWorld random_complex_init() const {
    g::GridWorld w(mSize, mSize, 2, 1, rand(), true);
    while (not w.is_solvable()){
      w = g::GridWorld(mSize, mSize, 2, 1, rand(), true);
    }
    return w;
  }

  g::GridWorld random_repeated_complex_init() const {
    if (mRepeatCount == 0){
      mPrevSeed = s::rand();
      mRepeatCount = mRepeatTotalCount;
    }
    g::GridWorld w(mSize, mSize, 2, 1, mPrevSeed, true);
    while (not w.is_solvable()){
      mPrevSeed = s::rand();
      w = g::GridWorld(mSize, mSize, 2, 1, mPrevSeed, true);
    }
    mRepeatCount--;
    return w;
  }

  void recursive_maze_generator_helper(g::GridWorld& w, const g::GridState& s, g::Pt pt, s::vector<bool>& visited) const {
    if (visited[pt_to_index(pt, mSize)] == true) return;

    visited[pt_to_index(pt, mSize)] = true;
    g::Pt next_pts[4U];
    g::Pt valid_pts[4U];
    g::all_next_pts(next_pts, pt, mSize);
    uint valid_sz = 0;
    for (uint i = 0; i < 4U; ++i)
      if (next_pts[i] != pt && s.get(next_pts[i]) == g::Obj::Empty && visited[pt_to_index(next_pts[i], mSize)] == false)
        valid_pts[valid_sz++] = next_pts[i];
    if (valid_sz < 2){
      for (uint i = 0; i < valid_sz; ++i)
        visited[pt_to_index(valid_pts[i], mSize)] = true;
      return;
    }
    uint wall_idx = rand() % valid_sz;
    w.set_wall(valid_pts[wall_idx]);
    visited[valid_pts[wall_idx].i * mSize + valid_pts[wall_idx].j] = true;
    if (valid_sz == 1) return;
    uint next_idx = wall_idx;
    while (next_idx == wall_idx)
      next_idx = rand() % valid_sz;
    recursive_maze_generator_helper(w, s, valid_pts[next_idx], visited);
  }

  void recursive_maze_generator(g::GridWorld& w, const g::GridState& s) const {
    s::vector<bool> visited(mSize * mSize, false);

    for (uint i = 0; i < visited.size(); ++i)
      if (visited[i] == false)
        recursive_maze_generator_helper(w, s, g::index_to_pt(i, mSize), visited);
  }

  g::GridWorld random_maze_init() const {
    assert(mSize > 1);

    g::GridWorld w(mSize, 0, 0, 0, rand(), false);
    w.remove_player();
    const g::GridState& ws = w.get_state();

    recursive_maze_generator(w, ws);

    // set player and goal locations
    do {
      int px = s::rand() % mSize, py = s::rand() % mSize,
          gx = s::rand() % mSize, gy = s::rand() % mSize;
      if (s::abs(px - gx) + s::abs(py - gy) - (int)(mSize + (mSize / 2))  < 0)
        continue;
      if (not w.set_player_location(g::Pt(px, py)))
        continue;
      if (not w.set_goal_location(g::Pt(gx, gy)))
        continue;
      break;
    } while (true);

    return w;
  }

  g::GridWorld random_maze_init_check() const {
    g::GridWorld w = random_maze_init();
    while (not w.is_solvable())
      w = random_maze_init();
    return w;
  }

  uint hash_world(const g::GridWorld& ins) const {
    uint hash = (ins.mPlayer.i << 8U) & (ins.mPlayer.j & 0xFFU);
    return hash;
  }

  bool is_move_in_history(const g::GridWorld& ins) const {
    uint ins_hash = hash_world(ins);
    if (mHistory.find(ins_hash) == mHistory.end()) return false;
    else                                           return true;
  }
public:
  GridEnv(
    uint sz,
    GridEnvMode mode,
    uint num_repeats = 0,
    bool step_discount = true,
    bool historical_discount = false,
    bool historical_termination = false):
    mSize(sz),
    mMode(mode),
    mRepeatTotalCount(num_repeats),
    mPrevSeed(0U),
    mRepeatCount(0U),
    mUseStepDiscount(step_discount),
    mUseHistoricalMoveDiscount(historical_discount),
    mTerminateOnHistoricalMove(historical_termination){
    switch (mMode){
    case GridEnvMode::RandomPlayerLocation:
    case GridEnvMode::RandomSimple:
    case GridEnvMode::RandomComplex:
    case GridEnvMode::RandomRepeatedComplex:
    case GridEnvMode::RandomMaze:
      s::srand(time(0));
    default:;
    }
  }

  g::GridWorld create() const {
    mHistory.clear();

    switch (mMode){
    case GridEnvMode::StaticSimple:
      return static_simple_init();
    case GridEnvMode::RandomPlayerLocation:
      return random_player_location_init();
    case GridEnvMode::RandomSimple:
      return random_simple_init();
    case GridEnvMode::RandomComplex:
      return random_complex_init();
    case GridEnvMode::RandomRepeatedComplex:
      return random_repeated_complex_init();
    case GridEnvMode::RandomMaze: //flawed, best with size <= 64
      return random_maze_init_check();
    default: assert(false);
    }
  }

  const g::GridState& get_state(const g::GridWorld& ins) const {
    return ins.get_state();
  }

  uint size() const {
    return mSize;
  }

  float get_reward(const g::GridWorld& ins) const {
    float reward = ins.get_reward();

    // we may shape the reward to explicitly discourage going backward
    // in this game a cheat to help improve result
    float history_penalty = -1.;
    if ((mUseHistoricalMoveDiscount || mTerminateOnHistoricalMove) &&
        is_move_in_history(ins)){
      history_penalty = ins.min_reward();
    }

    if (reward == 0){
      if (mUseStepDiscount)
        return s::max<float>(-1.F + history_penalty, ins.min_reward());
      else
        return s::max<float>(0.F + history_penalty, ins.min_reward());
    } else
      return s::min<float>(reward * 2.F, ins.max_reward());
  }

  float max_reward(const g::GridWorld& ins) const {
    return (float)ins.max_reward() * 2;
  }

  float min_reward(const g::GridWorld& ins) const {
    return (float)ins.min_reward() * 2;
  }

  void set_repeat_count(uint new_repeat){
    mRepeatTotalCount = new_repeat;
    mPrevSeed = rand();
  }

  void apply_action(g::GridWorld& ins, g::Action action) const {
    mHistory.insert(hash_world(ins));

    ins.move(action);
  }

  bool is_termination(g::GridWorld& ins) const {
    if (mTerminateOnHistoricalMove && is_move_in_history(ins))
      return true;
    else
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
    uint sz = env.size();
    mState = new float[sz * sz * 4];
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

  t::Tensor encode_state(const g::GridState& state, t::Device device) const {
    uint sz = mEnv.size();
    uint psz = sz * sz;
    uint encoded_state_sz = sz * sz * 4;
    //Plane1: player
    //Plane2: walls
    //Plane3: sinks
    //Plane4: goals
    for (uint i = 0; i < encoded_state_sz; ++i)
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
    t::Tensor m = t::from_blob(mState, {encoded_state_sz});
    if (device.type() == t::kCUDA){
      return m.to(device);
    } else {
      return m.clone();
    }
  }

  g::GridState decode_state(t::Tensor tensor) const {
    //TODO: see if I can read directly from GPU memory
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

  Dim state_size() const {
    uint sz = mEnv.size();
    return Dim(sz * sz * 4);
  }
};

//convolutional gridworld state encoder
class GridStateConvEncoder {
  const GridEnv& mEnv;
  mutable float* mState;
public:
  explicit GridStateConvEncoder(const GridEnv& env):
    mEnv(env), mState(nullptr) {
    mState = new float[env.size() * env.size() * 4];
  }
  GridStateConvEncoder(GridStateConvEncoder&& o): mEnv(o.mEnv), mState(o.mState) {
    o.mState = nullptr;
  }
  ~GridStateConvEncoder(){
    if (mState){
      delete[] mState;
      mState = nullptr;
    }
  }

  t::Tensor encode_state(const g::GridState& state, t::Device device) const {
    uint sz = mEnv.size();
    uint psz = sz * sz;
    uint full_size = state_size().flatten_size();
    //Plane1: player
    //Plane2: walls
    //Plane3: sinks
    //Plane4: goals
    for (uint i = 0; i < full_size; ++i)
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
    t::Tensor m = t::from_blob(mState, {4, sz, sz});
    if (device.type() == t::kCUDA){
      return m.to(device);
    } else {
      return m.clone();
    }
  }

  g::GridState decode_state(t::Tensor tensor) const {
    //TODO: see if I can read directly from GPU memory
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

  Dim state_size() const {
    uint sz = mEnv.size();
    return Dim(4, sz, sz);
  }
};

class GridActionEncoder {
  const GridEnv& mEnv;
  mutable float* mAction;
public:
  explicit GridActionEncoder(const GridEnv& env):
    mEnv(env), mAction(nullptr) {
    mAction = new float[(uint)g::Action::MAX];
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

  t::Tensor encode_action(g::Action action, t::Device device) const {
    uint sz = action_size();
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
    t::Tensor m = t::from_blob(mAction, {sz});
    if (device.type() == t::kCUDA){
      return m.to(device);
    } else {
      return m.clone();
    }
  }

  g::Action decode_action(t::Tensor tensor) const {
    uint max_idx = t::argmax(tensor).item().to<int>();

    assert(max_idx < action_size());

    switch (max_idx){
    case 0: return g::Action::UP;
    case 1: return g::Action::DN;
    case 2: return g::Action::LF;
    case 3: return g::Action::RT;
    default: assert(false);
    }
  }

  uint action_size() const {
    return (uint)g::Action::MAX;
  }
};

class SimpleQModelImpl : public t::nn::Module {
  t::nn::Linear l1, l2, l3;
public:
  SimpleQModelImpl(sint64 isz, sint64 l1sz, sint64 l2sz, sint64 osz):
    l1(register_module("l1", t::nn::Linear(isz, l1sz))),
    l2(register_module("l2", t::nn::Linear(l1sz, l2sz))),
    l3(register_module("l3", t::nn::Linear(l2sz, osz)))
  {}
  SimpleQModelImpl(const SimpleQModelImpl& o):
    l1(o.l1->options), l2(o.l2->options), l3(o.l3->options)
  {}

  t::Tensor forward(t::Tensor x){
    x = t::relu(l1(x));
    x = t::relu(l2(x));
    x = l3(x);
    return x;
  }
};
TORCH_MODULE(SimpleQModel);

class SimpleConvQModelImpl : public t::nn::Module {
  t::nn::Conv2d cl1, cl2;
  t::nn::Linear l3, l4;
  Dim           state_size;
public:
  SimpleConvQModelImpl(Dim isz, Dim cl1sz, Dim cl2sz, sint64 linsz, sint64 osz):
    cl1(register_module("cl1", t::nn::Conv2d(t::nn::Conv2dOptions(isz.x, cl1sz.x, cl1sz.y)))),
    cl2(register_module("cl2", t::nn::Conv2d(t::nn::Conv2dOptions(cl1sz.x, cl2sz.x, cl2sz.y)))),
    l3(register_module("l3", t::nn::Linear((isz.y - cl1sz.y - cl2sz.y + 2) * (isz.z - cl1sz.z - cl2sz.z + 2) * cl2sz.x, linsz))),
    l4(register_module("l4", t::nn::Linear(linsz, osz))),
    state_size(isz)
  {}
  SimpleConvQModelImpl(const SimpleConvQModelImpl& o):
    cl1(conv_options<2>(o.cl1->options)),
    cl2(conv_options<2>(o.cl2->options)),
    l3(o.l3->options),
    l4(o.l4->options),
    state_size(o.state_size)
  {}
  t::Tensor forward(t::Tensor x){
    if (x.dim() == 3)
      x = x.reshape({1, state_size.x, state_size.y, state_size.z});
    x = t::relu(cl1(x));
    x = t::relu(cl2(x));
    x = t::relu(l3(x.flatten(1, -1)));
    x = l4(x);
    return x;
  }
};
TORCH_MODULE(SimpleConvQModel);

class SimpleICMQModelImpl : public t::nn::Module {
  // share state encoder across all functions; trained only by inverse dynamics module;
  // this way the state encoder will only encode state that's going to affect the agent's
  // action
  t::nn::Conv2d cl1, cl2, sec1, sec2;
  t::nn::Linear sel1, l3, l4, fdl1, idl1;
  Dim           state_size;
public:
  SimpleICMQModelImpl(Dim ssz, Dim cl1sz, Dim cl2sz, Dim sec1sz, Dim sec2sz, sint64 sfsz, sint64 l1sz, sint64 asz):
    cl1(register_module("cl1", t::nn::Conv2d(t::nn::Conv2dOptions(ssz.x, cl1sz.x, cl1sz.y)))),
    cl2(register_module("cl2", t::nn::Conv2d(t::nn::Conv2dOptions(cl1sz.x, cl2sz.x, cl2sz.y)))),
    sec1(register_module("sec1", t::nn::Conv2d(t::nn::Conv2dOptions(ssz.x, sec1sz.x, sec1sz.y)))),
    sec2(register_module("sec2", t::nn::Conv2d(t::nn::Conv2dOptions(sec1sz.x, sec2sz.x, sec2sz.y)))),
    sel1(register_module("sel1", t::nn::Linear((ssz.y - sec1sz.y - sec2sz.y + 2) * (ssz.z - sec1sz.z - sec2sz.z + 2) * sec2sz.x, sfsz))),
    l3(register_module("l3", t::nn::Linear((ssz.y - cl1sz.y - cl2sz.y + 2) * (ssz.z - cl1sz.z - cl2sz.z + 2) * cl2sz.x, l1sz))),
    l4(register_module("l4", t::nn::Linear(l1sz, asz))),
    fdl1(register_module("fdl1", t::nn::Linear(sfsz + asz, sfsz))),
    idl1(register_module("idl1", t::nn::Linear(sfsz + sfsz, asz))),
    state_size(ssz)
  {}
  SimpleICMQModelImpl(const SimpleICMQModelImpl& o):
    cl1(conv_options<2>(o.cl1->options)),
    cl2(conv_options<2>(o.cl2->options)),
    sec1(conv_options<2>(o.sec1->options)),
    sec2(conv_options<2>(o.sec2->options)),
    sel1(o.sel1->options),
    l3(o.l3->options),
    l4(o.l4->options),
    fdl1(o.fdl1->options),
    idl1(o.idl1->options),
    state_size(o.state_size)
  {}
  t::Tensor featurize_state(t::Tensor state){
    if (state.dim() == 3)
      state = state.reshape({1, state_size.x, state_size.y, state_size.z});
    state = t::relu(sec1(state));
    state = t::relu(sec2(state));
    state = t::relu(sel1(state.flatten(1, -1)));
    return state;
  }
  t::Tensor icm_forward_dynamics(t::Tensor state, t::Tensor action){
    state = featurize_state(state);
    t::Tensor x = t::cat({state, action}, -1);
    //do not use forward dynamics to learn state feature
    x = t::relu(fdl1(x.detach()));
    return x;
  }
  t::Tensor icm_inverse_dynamics(t::Tensor state, t::Tensor nstate){
    state  = featurize_state(state);
    nstate = featurize_state(nstate);
    t::Tensor x = t::cat({state, nstate}, -1);
    x = t::softmax(idl1(x), -1);
    return x;
  }
  t::Tensor forward(t::Tensor state){
    if (state.dim() == 3)
      state = state.reshape({1, state_size.x, state_size.y, state_size.z});
    state = t::relu(cl1(state));
    state = t::relu(cl2(state));
    state = t::relu(l3(state.flatten(1, -1)));
    state = l4(state);
    return state;
  }
};
TORCH_MODULE(SimpleICMQModel);

class SimplePolicyModelImpl : public t::nn::Module {
  t::nn::Linear l1, l2, l3;
public:
  SimplePolicyModelImpl(sint64 isz, sint64 l1sz, sint64 l2sz, sint64 osz):
    l1(register_module("l1", t::nn::Linear(isz, l1sz))),
    l2(register_module("l2", t::nn::Linear(l1sz, l2sz))),
    l3(register_module("l3", t::nn::Linear(l2sz, osz)))
  {}
  t::Tensor forward(t::Tensor x){
    x = t::relu(l1(x));
    x = t::relu(l2(x));
    x = t::softmax(l3(x), -1);
    return x;
  }
};
TORCH_MODULE(SimplePolicyModel);

struct ACTensor {
  t::Tensor actor_out;
  t::Tensor critic_out;

  ACTensor() = default;
  ACTensor(t::Tensor ao, t::Tensor co): actor_out(ao), critic_out(co) {}
};

class SimpleActorCriticModelImpl : public t::nn::Module {
  t::nn::Linear l1, l2, l3a, l3c, l4c;
public:
  SimpleActorCriticModelImpl(sint64 isz, sint64 l1sz, sint64 l2sz, sint64 l3csz, sint64 oasz, sint64 ocsz):
    l1(register_module("l1", t::nn::Linear(isz, l1sz))),
    l2(register_module("l2", t::nn::Linear(l1sz, l2sz))),
    l3a(register_module("l3a", t::nn::Linear(l2sz, oasz))),
    l3c(register_module("l3c", t::nn::Linear(l2sz, l3csz))),
    l4c(register_module("l4c", t::nn::Linear(l3csz, ocsz)))
  {}
  t::Tensor actor_forward(t::Tensor x){
    x = t::relu(l1(x));
    x = t::relu(l2(x));
    t::Tensor ao = t::softmax(l3a(x), -1);
    return ao;
  }
  t::Tensor critic_forward(t::Tensor x){
    x = t::relu(l1(x));
    x = t::relu(l2(x));
    t::Tensor co = t::relu(l3c(x.detach())); //no backprop
    co = t::tanh(l4c(co));
    co = co.squeeze();
    return co;
  }
  ACTensor forward(t::Tensor x){
    x = t::relu(l1(x));
    x = t::relu(l2(x));
    t::Tensor ao = t::softmax(l3a(x), -1);
    t::Tensor co = t::relu(l3c(x.detach())); //no backprop
    co = t::tanh(l4c(co));
    co = co.squeeze();
    return ACTensor(ao, co);
  }
};
TORCH_MODULE(SimpleActorCriticModel);

class SimpleDistQModelImpl : public t::nn::Module {
  t::nn::Linear l1, l2, l3;
  sint64 action_sz;
public:
  SimpleDistQModelImpl(sint64 isz, sint64 l1sz, sint64 l2sz, sint64 action_sz, sint64 distout_sz):
    l1(register_module("l1", t::nn::Linear(isz, l1sz))),
    l2(register_module("l2", t::nn::Linear(l1sz, l2sz))),
    l3(register_module("l3", t::nn::Linear(l2sz, action_sz * distout_sz))),
    action_sz(action_sz)
  {}
  SimpleDistQModelImpl(const SimpleDistQModelImpl& o):
    l1(o.l1->options), l2(o.l2->options), l3(o.l3->options), action_sz(o.action_sz)
  {}

  t::Tensor forward(t::Tensor x){
    x = t::relu(l1(x));
    x = t::relu(l2(x));
    x = l3(x);
    if (x.dim() == 1)
      x = x.reshape({action_sz, -1});
    else
      x = x.reshape({x.size(0), action_sz, -1});
    x = t::softmax(x, -1);
    return x;
  }
};
TORCH_MODULE(SimpleDistQModel);

class SimplePPOICMModelImpl : public t::nn::Module {
  t::nn::Conv2d cl1, cl2, sec1, sec2;
  t::nn::Linear sel1, l3a, l4a, l3c, l4c, fdl1, idl1;
  Dim           state_size;

  t::Tensor conv_state(t::Tensor state){
    if (state.dim() == 3)
      state = state.reshape({1, state_size.x, state_size.y, state_size.z});
    state = t::relu(cl1(state));
    state = t::relu(cl2(state));
    return state.flatten(1, -1);
  }
public:
  SimplePPOICMModelImpl(Dim ssz, Dim cl1sz, Dim cl2sz, Dim sec1sz, Dim sec2sz, sint64 sfsz, sint64 alsz, sint64 clsz, sint64 asz):
    cl1(register_module("cl1", t::nn::Conv2d(t::nn::Conv2dOptions(ssz.x, cl1sz.x, cl1sz.y)))),
    cl2(register_module("cl2", t::nn::Conv2d(t::nn::Conv2dOptions(cl1sz.x, cl2sz.x, cl2sz.y)))),
    sec1(register_module("sec1", t::nn::Conv2d(t::nn::Conv2dOptions(ssz.x, sec1sz.x, sec1sz.y)))),
    sec2(register_module("sec2", t::nn::Conv2d(t::nn::Conv2dOptions(sec1sz.x, sec2sz.x, sec2sz.y)))),
    sel1(register_module("sel1", t::nn::Linear((ssz.y - sec1sz.y - sec2sz.y + 2) * (ssz.z - sec1sz.z - sec2sz.z + 2) * sec2sz.x, sfsz))),
    l3a(register_module("l3a", t::nn::Linear((ssz.y - cl1sz.y - cl2sz.y + 2) * (ssz.z - cl1sz.z - cl2sz.z + 2) * cl2sz.x, alsz))),
    l4a(register_module("l4a", t::nn::Linear(alsz, asz))),
    l3c(register_module("l3c", t::nn::Linear((ssz.y - cl1sz.y - cl2sz.y + 2) * (ssz.z - cl1sz.z - cl2sz.z + 2) * cl2sz.x, clsz))),
    l4c(register_module("l4c", t::nn::Linear(clsz, 1))),
    fdl1(register_module("fdl1", t::nn::Linear(sfsz + asz, sfsz))),
    idl1(register_module("idl1", t::nn::Linear(sfsz + sfsz, asz))),
    state_size(ssz)
  {}
  SimplePPOICMModelImpl(const SimplePPOICMModelImpl& o):
    cl1(conv_options<2>(o.cl1->options)),
    cl2(conv_options<2>(o.cl2->options)),
    sec1(conv_options<2>(o.sec1->options)),
    sec2(conv_options<2>(o.sec2->options)),
    sel1(o.sel1->options),
    l3a(o.l3a->options),
    l4a(o.l4a->options),
    l3c(o.l3c->options),
    l4c(o.l4c->options),
    fdl1(o.fdl1->options),
    idl1(o.idl1->options),
    state_size(o.state_size)
  {}
  t::Tensor featurize_state(t::Tensor state){
    if (state.dim() == 3)
      state = state.reshape({1, state_size.x, state_size.y, state_size.z});
    state = t::relu(sec1(state));
    state = t::relu(sec2(state));
    state = t::relu(sel1(state.flatten(1, -1)));
    return state;
  }
  t::Tensor icm_forward_dynamics(t::Tensor state, t::Tensor action){
    state = featurize_state(state);
    t::Tensor x = t::cat({state, action}, -1);
    //do not use forward dynamics to learn state features
    x = t::relu(fdl1(x.detach()));
    return x;
  }
  t::Tensor icm_inverse_dynamics(t::Tensor state, t::Tensor nstate){
    state = featurize_state(state);
    nstate = featurize_state(nstate);
    t::Tensor x = t::cat({state, nstate}, -1);
    x = t::softmax(idl1(x), -1);
    return x;
  }
  t::Tensor actor_forward(t::Tensor state){
    state = conv_state(state);
    t::Tensor ao = t::relu(l3a(state));
    ao = t::softmax(l4a(ao), -1);
    return ao;
  }
  t::Tensor critic_forward(t::Tensor state){
    state = conv_state(state);
    //do not use value function to train policy convolution
    t::Tensor co = t::relu(l3c(state.detach()));
    co = t::tanh(l4c(co));
    co = co.squeeze();
    return co;
  }
  ACTensor forward(t::Tensor state){
    state = conv_state(state);
    t::Tensor ao = t::relu(l3a(state));
    ao = t::softmax(l4a(ao), -1);
    //do not use value function to train policy convolution
    t::Tensor co = t::relu(l3c(state.detach()));
    co = t::tanh(l4c(co));
    co = co.squeeze();
    return ACTensor(ao, co);
  }
};
TORCH_MODULE(SimplePPOICMModel);

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
