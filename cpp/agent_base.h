#ifndef RLGAMES_AGENT_BASE
#define RLGAMES_AGENT_BASE

#include <cassert>

#include <type_alias.h>
#include <types.h>

namespace rlgames {

template <typename Board, typename GameState, typename Sub>
struct AgentBase {
  Move select_move(const GameState& gs){
    return static_cast<Sub*>(this)->select_move(gs);
  }
};

//heuristic helper functions
template <typename Board>
struct IsPointAnEye {
  bool operator()(const Board&, Pt, Player){
    return false;
  }
};

template <ubyte SZ> struct GoBoard;

template <ubyte SZ>
struct IsPointAnEye<GoBoard<SZ>> {
  bool operator()(const GoBoard<SZ>& board, Pt pt, Player player){
    assert(board.is_on_grid(pt));

    //if point is not empty, it's not an eye
    if (board.get(pt) != Player::Unknown) return false;

    for (Pt neighbour : neighbours(pt)){
      if (not board.is_on_grid(neighbour)) continue;

      if (board.get(neighbour) != player)
        return false;
    }

    uint friendly_corner_count = 0;
    uint off_board_count = 0;
    for (Pt corner : corners(pt)){
      if (board.is_on_grid(corner)){
        if (board.get(corner) == player)
          friendly_corner_count++;
      } else
        off_board_count++;
    }
    if (off_board_count > 0)
      return off_board_count + friendly_corner_count == 4;
    else
      return friendly_corner_count >= 3;
  }
};

} // rlgames

#endif//RLGAMES_AGENT_BASE
