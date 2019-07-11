#ifndef RLGAMES_RANDOM_AGENT
#define RLGAMES_RANDOM_AGENT

#include <ctime>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

#include <types.h>
#include <agent_base.h>

namespace s = std;

namespace rlgames {

template <typename Board, typename GameState>
class RandomAgent : public AgentBase<Board, GameState, RandomAgent<Board, GameState>> {
  s::vector<udyte> mCache;

  void initialize_cache(size_t sz){
    mCache.resize(sz);
    s::iota(mCache.begin(), mCache.end(), 0);
  }
public:
  RandomAgent(){
    s::srand(unsigned(s::time(0)));
  }
  Move select_move(GameState& gs){
    if (mCache.size() == 0){
      size_t sz = gs.board().size();
      initialize_cache(sz * sz);
    }
    s::random_shuffle(s::begin(mCache), s::end(mCache), [](uint i){ return s::rand() % i; }); //TODO: avoid the %
    for (udyte index : mCache){
      Pt pt = point<Board::SIZE>(index);
      Move m(M::Play, pt);
      if (gs.is_valid_move(m) && (not is_point_an_eye<Board>(gs.board(), pt, gs.next_player())))
        return m;
    }
    return Move(M::Pass);
  }
};

} // rlgames

#endif//RLGAMES_RANDOM_AGENT
