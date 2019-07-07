#ifndef RLGAMES_TTT_TYPES
#define RLGAMES_TTT_TYPES

#include <cassert>
#include <array>
#include <vector>

#include <type_alias.h>
#include <types.h>

namespace s = std;

namespace rlgames {

template <ubyte SZ> struct TTTBoard_;

template <>
struct TTTBoard_<3> {
  static constexpr size_t SZ = 3;
private:
  s::array<Player, SZ * SZ> mBoard;
public:
  TTTBoard_(){
    memset(&mBoard[0], (ubyte)Player::Unknown, SZ * SZ * sizeof(ubyte));
  }
  TTTBoard_(const TTTBoard_& o): mBoard(o.mBoard) {}
  TTTBoard_& operator=(const TTTBoard_& o){
    mBoard = o.mBoard;
    return *this;
  }

  bool is_on_grid(Pt pt) const {
    if (pt.r < SZ && pt.c < SZ) return true;
    else                        return false;
  }
  Player get(Pt pt) const {
    assert(is_on_grid(pt));

    return mBoard[index<SZ>(pt)];
  }
  void place_stone(Player player, Pt pt){
    assert(is_on_grid(pt));
    assert(get(pt) == Player::Unknown);

    mBoard[index<SZ>(pt)] = player;
  }
  size_t size() const {
    return SZ;
  }
};

using TTTBoard = TTTBoard_<3>;

class TTTGameState {
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
      drsum += mBoard.get(Pt(2 - i, 2 - i)) == player;
    }
    if (rsum == 3 || csum == 3 || dfsum == 3 || drsum == 3) return true;
    else                                                    return false;
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

  bool is_over() const {
    switch (mPMove.mty){
    case M::Unknown:
      return false;
    case M::Pass:
    case M::Resign:
      return true;
    case M::Play:
      return is_connected();
    default: assert(false);
    }
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
    for (size_t i = 0; i < TTTBoard::SZ * TTTBoard::SZ; ++i){
      if (mBoard.get(point<TTTBoard::SZ>(i)) == Player::Unknown)
        ret.push_back(Move(M::Play, point<TTTBoard::SZ>(i)));
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
