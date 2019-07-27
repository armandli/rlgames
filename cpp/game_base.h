#ifndef RLGAMES_GAME_BASE
#define RLGAMES_GAME_BASE

#include <ostream>

#include <type_alias.h>
#include <types.h>

namespace rlgames {

template <typename Sub>
struct Board {
  bool is_on_grid(Pt pt) const {
    return static_cast<Sub*>(this)->is_on_grid(pt);
  }
  Player get(Pt pt) const {
    return static_cast<Sub*>(this)->get(pt);
  }
  void place_stone(Player player, Pt pt){
    static_cast<Sub*>(this)->place_stone(player, pt);
  }
  //TODO: make this static method
  size_t size() const {
    return static_cast<Sub*>(this)->size();
  }
  s::ostream& print(s::ostream& out) const {
    return static_cast<Sub*>(this)->print(out);
  }
};

template <typename Sub>
s::ostream& operator<<(s::ostream& out, const Board<Sub>& board){
  return board.print(out);
}

template <typename Board, typename Sub>
struct GameState {
  const Board& board() const {
    return static_cast<Sub*>(this)->board();
  }
  Player next_player() const {
    return static_cast<Sub*>(this)->next_player();
  }
  Move previous_move() const {
    return static_cast<Sub*>(this)->previous_move();
  }
  bool is_over() const {
    return static_cast<Sub*>(this)->is_over();
  }
  bool is_valid_move(Move m) const {
    return static_cast<Sub*>(this)->is_valid_move(m);
  }
  s::vector<Move> legal_moves() const {
    return static_cast<Sub*>(this)->legal_moves();
  }
  Player winner() const {
    return static_cast<Sub*>(this)->winner();
  }
  GameState& apply_move(Move move){
    return static_cast<Sub*>(this)->apply_move(move);
  }
};

} // rlgames

#endif//RLGAMES_GAME_BASE
