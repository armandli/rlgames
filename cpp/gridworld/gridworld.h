#ifndef GRIDWORLD_GRIDWORLD
#define GRIDWORLD_GRIDWORLD

#include <cassert>
#include <cstdlib>
#include <vector>
#include <queue>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <ostream>

#include <type_alias.h>

namespace s = std;

namespace gridworld {

struct Pt {
  uint i, j;

  Pt() = default;
  Pt(uint i, uint j): i(i), j(j) {}
  Pt(const Pt& o) = default;
  //Pt& operator=(const Pt& o) = default; //TODO
};

bool operator==(Pt a, Pt b){
  return a.i == b.i && a.j == b.j;
}

bool operator!=(Pt a, Pt b){
  return not operator==(a, b);
}

bool operator<(Pt a, Pt b){
  if (a.i < b.i)               return true;
  if (a.i == b.i && a.j < b.j) return true;
  return false;
}

Pt rand_coord(uint sz){
  return Pt(s::rand() % sz, s::rand() % sz);
}

enum class Action: ubyte {
  UP = 0,
  DN,
  LF,
  RT,
  MAX,
};

void all_next_pts(Pt* out, Pt p, uint size){
  for (ubyte i = (ubyte)Action::UP; i < (ubyte)Action::MAX; ++i){
    Pt m = p;
    switch (i){
    case (ubyte)Action::UP:
      m.i = s::min(m.i - 1, size);
      m.i %= size;
      break;
    case (ubyte)Action::DN:
      m.i = s::min(m.i + 1, size - 1);
      break;
    case (ubyte)Action::LF:
      m.j = s::min(m.j - 1, size);
      m.j %= size;
      break;
    case (ubyte)Action::RT:
      m.j = s::min(m.j + 1, size - 1);
      break;
    default: assert(false);
    }
    out[i] = m;
  }
}

enum class Obj: ubyte {
  Empty,
  Player,
  Wall,
  Sink,
  Goal,
  PS,    //Player and Sink overlap
  PG,    //Player and Goal overlap
};

class GridState {
  uint           mSize;
  s::vector<Obj> mMap;

  uint to_index(Pt c) const {
    return c.i * mSize + c.j;
  }
public:
  explicit GridState(uint sz): mSize(sz), mMap(sz * sz, Obj::Empty) {}

  GridState(const GridState& o) = default;
  //GridState& operator=(const GridState& o) = default; //TODO

  void set_cell(Obj obj, Pt c){
    uint idx = to_index(c);
    assert(idx < mSize * mSize);

    if (obj == Obj::Player){
      assert(mMap[idx] != Obj::Wall);
      if (mMap[idx] == Obj::Sink)
        mMap[idx] = Obj::PS;
      else if (mMap[idx] == Obj::Goal)
        mMap[idx] = Obj::PG;
      else
        mMap[idx] = Obj::Player;
    } else {
      assert(mMap[idx] == Obj::Empty);
      mMap[idx] = obj;
    }
  }
  void set_cell(Obj obj, uint idx){
    assert(idx < mSize * mSize);

    if (obj == Obj::Player){
      assert(mMap[idx] != Obj::Wall);
      if (mMap[idx] == Obj::Sink)
        mMap[idx] = Obj::PS;
      else if (mMap[idx] == Obj::Goal)
        mMap[idx] = Obj::PG;
      else
        mMap[idx] = Obj::Player;
    } else {
      assert(mMap[idx] == Obj::Empty);
      mMap[idx] = obj;
    }
  }
  void clear_cell(Pt c){
    uint idx = to_index(c);
    assert(idx < mSize * mSize);

    mMap[idx] = Obj::Empty;
  }
  Obj get(Pt c) const {
    assert(c.i < mSize && c.j < mSize);
    return mMap[to_index(c)];
  }
  Obj get(uint idx) const {
    assert(idx < mSize * mSize);
    return mMap[idx];
  }
  bool is_complete() const {
    for (Obj obj : mMap)
      switch (obj){
      case Obj::Player: return false;
      case Obj::PS:     return true;
      case Obj::PG:     return true;
      default:;         //do nothing
      }
    return false;
  }

  bool is_solvable(Pt start) const {
    s::vector<bool> seen(mMap.size(), false);
    s::queue<Pt> q; q.push(start);
    Pt next_moves[(uint)Action::MAX];

    while (not q.empty()){
      Pt pt = q.front(); q.pop();
      uint idx = to_index(pt);
      switch (mMap[idx]){
      case Obj::Goal:
      case Obj::PG:
        return true;
      default:; //do nothing
      }
      seen[idx] = true;
      all_next_pts(next_moves, pt, mSize);
      for (uint i = 0; i < mSize; ++i){
        uint nidx = to_index(next_moves[i]);
        if (seen[nidx]) continue;
        switch (mMap[nidx]){
        case Obj::Empty:
        case Obj::Player:
        case Obj::Goal:
        case Obj::PG:
          q.push(next_moves[i]);
          break;
        case Obj::Wall:
        case Obj::Sink:
        case Obj::PS:
          continue;
        }
      }
    }
    return false;
  }

  s::ostream& print(s::ostream& out) const {
    for (uint i = 0; i < mSize; ++i)
      out << '-';
    out << '\n';

    for (uint i = 0; i < mSize; ++i){
      for (uint j = 0; j < mSize; ++j)
        switch (get(Pt(i, j))){
        case Obj::Empty:
          out << ' ';
          break;
        case Obj::Player:
          out << 'P';
          break;
        case Obj::Wall:
          out << 'W';
          break;
        case Obj::Sink:
          out << '-';
          break;
        case Obj::Goal:
          out << '+';
          break;
        case Obj::PS:
          out << 'S';
          break;
        case Obj::PG:
          out << 'G';
          break;
        default: assert(false);
        }
      out << '\n';
    }

    for (uint i = 0; i < mSize; ++i)
      out << '-';
    out << s::endl;
    return out;
  }
};

//TODO: abstract a set interface so we can switch to different set implementations
template <typename T>
using Set = s::set<T>;

class GridWorld {
  uint      mStepCount;
  uint      mSize;
  GridState mState;
  Pt        mPlayer;
  Set<Pt>   mWalls;
  Set<Pt>   mSinks;
  Set<Pt>   mGoals;
  bool      mUseStepCount;

  void initialize_state(){
    mState.set_cell(Obj::Player, mPlayer);
    for (Pt c : mWalls)
      mState.set_cell(Obj::Wall, c);
    for (Pt c : mSinks)
      mState.set_cell(Obj::Sink, c);
    for (Pt c : mGoals)
      mState.set_cell(Obj::Goal, c);
  }

  void initialize(uint num_walls, uint num_sinks, uint num_goals, uint seed){
    s::srand(seed);
    mPlayer = rand_coord(mSize);
    while (mWalls.size() < num_walls){
      Pt c = rand_coord(mSize);
      if (c == mPlayer) continue;
      mWalls.insert(c);
    }
    while (mSinks.size() < num_sinks){
      Pt c = rand_coord(mSize);
      if (c == mPlayer) continue;
      decltype(mWalls.begin()) witer = mWalls.find(c);
      if (witer != mWalls.end()) continue;
      mSinks.insert(c);
    }
    while (mGoals.size() < num_goals){
      Pt c = rand_coord(mSize);
      if (c == mPlayer) continue;
      decltype(mWalls.begin()) witer = mWalls.find(c);
      if (witer != mWalls.end()) continue;
      decltype(mSinks.begin()) siter = mSinks.find(c);
      if (siter != mSinks.end()) continue;
      mGoals.insert(c);
    }

    initialize_state();
  }
public:
  GridWorld(uint sz, uint num_walls, uint num_sinks, uint num_goals, uint seed = 0U, bool discount_steps = false):
    mStepCount(0), mSize(sz), mState(sz), mUseStepCount(discount_steps) {
    initialize(num_walls, num_sinks, num_goals, seed);
  }
  GridWorld(const GridWorld& o) = default;
  //GridWorld& operator=(const GridWorld& o) = default; //TODO

  const GridState& get_state() const {
    return mState;
  }

  uint size() const {
    return mSize;
  }

  s::ostream& print(s::ostream& out) const {
    return mState.print(out);
  }

  bool is_complete() const {
    decltype(mSinks.begin()) siter = mSinks.find(mPlayer);
    if (siter != mSinks.end()) return true;
    decltype(mGoals.begin()) giter = mGoals.find(mPlayer);
    if (giter != mGoals.end()) return true;
    return false;
  }

  bool is_solvable() const {
    return mState.is_solvable(mPlayer);
  }

  int get_reward() const {
    int max_reward = mSize * mSize;

    decltype(mSinks.begin()) siter = mSinks.find(mPlayer);
    if (siter != mSinks.end()) return max_reward * -1;
    decltype(mGoals.begin()) giter = mGoals.find(mPlayer);
    if (mUseStepCount){
      if (giter != mGoals.end()) return max_reward - mStepCount;
    } else {
      if (giter != mGoals.end()) return max_reward;
    }
    return 0;
  }

  uint step_count() const {
    return mStepCount;
  }

  void move(Action action){
    mStepCount++;

    Pt nc = mPlayer;
    switch (action){
    case Action::UP:
      nc.i = s::min(nc.i - 1, mSize);
      nc.i = nc.i % mSize;
      break;
    case Action::DN:
      nc.i = s::min(nc.i + 1, mSize - 1);
      //nc.i = nc.i % mSize;
      break;
    case Action::LF:
      nc.j = s::min(nc.j - 1, mSize);
      nc.j = nc.j % mSize;
      break;
    case Action::RT:
      nc.j = s::min(nc.j + 1, mSize - 1);
      //nc.j = nc.j % mSize;
      break;
    default: assert(false);
    }
    decltype(mWalls.begin()) witer = mWalls.find(nc);
    if (witer != mWalls.end())
      return;

    mState.clear_cell(mPlayer);
    mState.set_cell(Obj::Player, nc);
    mPlayer = nc;
  }

  bool set_player_location(Pt c){
    decltype(mWalls.begin()) witer = mWalls.find(c);
    if (witer == mWalls.end()) return false;
    decltype(mSinks.begin()) siter = mSinks.find(c);
    if (siter == mSinks.end()) return false;
    decltype(mGoals.begin()) giter = mGoals.find(c);
    if (giter == mGoals.end()) return false;
    mState.clear_cell(mPlayer);
    mPlayer = c;
    mState.set_cell(Obj::Player, mPlayer);
    return true;
  }
};

} // gridworld

#endif//GRIDWORLD_GRIDWORLD
