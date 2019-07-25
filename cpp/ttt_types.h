#ifndef RLGAMES_TTT_TYPES
#define RLGAMES_TTT_TYPES

#include <cassert>
#include <cstring>
#include <array>
#include <vector>
#include <ostream>

#include <type_alias.h>
#include <types.h>
#include <game_base.h>

namespace s = std;

namespace rlgames {

template <ubyte SZ> struct TTTBoard_;

template <>
struct TTTBoard_<3> : Board<TTTBoard_<3>> {
  static constexpr uint SIZE = 3;
  static constexpr uint IZ = SIZE * SIZE;
private:
  s::array<Player, IZ> mBoard;
public:
  TTTBoard_(){
    s::memset(&mBoard[0], (ubyte)Player::Unknown, IZ * sizeof(ubyte));
  }
  TTTBoard_(const TTTBoard_& o): mBoard(o.mBoard) {}
  TTTBoard_& operator=(const TTTBoard_& o){
    mBoard = o.mBoard;
    return *this;
  }
  TTTBoard_(TTTBoard_&& o) noexcept : mBoard(s::move(o.mBoard)) {}
  TTTBoard_& operator=(TTTBoard_&& o) noexcept {
    mBoard = s::move(o.mBoard);
    return *this;
  }

  bool is_on_grid(Pt pt) const {
    if (pt.r < SIZE && pt.c < SIZE) return true;
    else                            return false;
  }
  Player get(Pt pt) const {
    assert(is_on_grid(pt));

    return mBoard[index<SIZE>(pt)];
  }
  void place_stone(Player player, Pt pt){
    assert(is_on_grid(pt));
    assert(get(pt) == Player::Unknown);

    mBoard[index<SIZE>(pt)] = player;
  }
  size_t size() const {
    return SIZE;
  }
  s::ostream& print(s::ostream& out) const {
    char bchar = 'x';
    char wchar = 'o';

    for (ubyte i = 0; i < SIZE; ++i){
      out << (int)i << ' ';
      for (ubyte j = 0; j < SIZE; ++j){
        Player color = get(Pt(i, j));
        switch (color){
        case Player::Black: out << bchar; break;
        case Player::White: out << wchar; break;
        case Player::Unknown: out << ' '; break;
        default: assert(false);
        }
        out << ' ';
      }
      out << '\n';
    }
    out << "  A B C\n";
    return out;
  }
};

s::ostream& operator<<(s::ostream& out, const TTTBoard_<3>& board){
  return board.print(out);
}

using TTTBoard = TTTBoard_<3>;

struct TTTGameState : public GameState<TTTBoard, TTTGameState> {
  static constexpr uint SIZE = 3;
  static constexpr uint IZ = 9;
private:
  TTTBoard mBoard;
  Player   mNPlayer;
  Move     mPMove;

protected:
  bool is_connected() const {
    assert(mPMove.mty == M::Play);
    Pt pt = mPMove.mpt;
    Player player = other_player(mNPlayer);
    size_t rsum = 0, csum = 0, dfsum = 0, drsum = 0;
    for (size_t i = 0; i < 3; ++i){
      rsum += mBoard.get(Pt(i, pt.c)) == player;
      csum += mBoard.get(Pt(pt.r, i)) == player;
      dfsum += mBoard.get(Pt(i, i)) == player;
      drsum += mBoard.get(Pt(i, 2 - i)) == player;
    }
    if (rsum == 3 || csum == 3 || dfsum == 3 || drsum == 3) return true;
    else                                                    return false;
  }
  uint empty_positions() const {
    uint count = 0;
    for (uint i = 0; i < IZ; ++i)
      if (mBoard.get(point<SIZE>(i)) == Player::Unknown)
        count++;
    return count;
  }
public:
  TTTGameState(): mBoard(), mNPlayer(Player::Black), mPMove(M::Unknown) {}
  TTTGameState(const TTTBoard& board, Player nplayer, Move move):
    mBoard(board), mNPlayer(nplayer), mPMove(move) {}
  TTTGameState(const TTTGameState& o):
    mBoard(o.mBoard), mNPlayer(o.mNPlayer), mPMove(o.mPMove) {}
  TTTGameState& operator=(const TTTGameState& o){
    mBoard = o.mBoard;
    mNPlayer = o.mNPlayer;
    mPMove = o.mPMove;
    return *this;
  }
  TTTGameState(TTTGameState&& o) noexcept :
    mBoard(s::move(o.mBoard)), mNPlayer(o.mNPlayer), mPMove(o.mPMove) {}
  TTTGameState& operator=(TTTGameState&& o) noexcept {
    mBoard = s::move(o.mBoard);
    mNPlayer = o.mNPlayer;
    mPMove = o.mPMove;
    return *this;
  }

  const TTTBoard& board() const { return mBoard; }
  Player next_player() const { return mNPlayer; }
  Move previous_move() const { return mPMove; }

  bool is_over() const {
    switch (mPMove.mty){
    case M::Unknown:
      return false;
    case M::Pass:
    case M::Resign:
      return true;
    case M::Play:
      if (is_connected()) return true;
      break;
    default: assert(false);
    }
    return empty_positions() == 0U;
  }

  bool is_valid_move(Move move) const {
    if (is_over())                                    return false;
    if (move.mty == M::Pass || move.mty == M::Resign) return true;
    return mBoard.get(move.mpt) == Player::Unknown;
  }

  s::vector<Move> legal_moves() const {
    if (is_over()) return s::vector<Move>();

    s::vector<Move> ret;
    ret.reserve(10);
    for (size_t i = 0; i < TTTBoard::IZ; ++i){
      if (mBoard.get(point<TTTBoard::SIZE>(i)) == Player::Unknown)
        ret.push_back(Move(M::Play, point<TTTBoard::SIZE>(i)));
    }
    ret.push_back(Move(M::Pass));
    return ret;
  }

  Player winner() const {
    switch (mPMove.mty){
    case M::Unknown:
      return Player::Unknown;
    case M::Pass:
    case M::Resign:
      return mNPlayer;
    case M::Play:
      if (is_connected()) return other_player(mNPlayer);
      else                return Player::Unknown;
    default: assert(false);
    }
  }

  TTTGameState& apply_move(Move move){
    assert(is_over() == false);

    if (move.mty == M::Play)
      mBoard.place_stone(mNPlayer, move.mpt);
    mPMove = move;
    mNPlayer = other_player(mNPlayer);
    return *this;
  }
};

} // rlgames

#endif//RLGAMES_TTT_TYPES
