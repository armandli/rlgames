#ifndef RLGAMES_TYPES
#define RLGAMES_TYPES

#include <cassert>
#include <array>
#include <ostream>
#include <istream>

#include <type_alias.h>

namespace rlgames {

namespace s = std;

struct Pt {
  ubyte r, c;

  Pt() = default;
  Pt(ubyte r, ubyte c): r(r), c(c) {}
};

bool operator==(Pt a, Pt b){
  return a.r == b.r && a.c == b.c;
}

bool operator!=(Pt a, Pt b){
  return not operator==(a, b);
}

using Neighbours = s::array<Pt, 4>; 

template <uint SZ>
[[gnu::always_inline]] uint index(Pt p){
  return p.r * SZ + p.c;
}

template <uint SZ>
[[gnu::always_inline]] Pt point(uint idx){
  return Pt(idx / SZ, idx % SZ);
}

[[gnu::always_inline]] Neighbours neighbours(Pt p){
  return s::array<Pt, 4>{Pt(p.r - 1, p.c), Pt(p.r, p.c - 1), Pt(p.r + 1, p.c), Pt(p.r, p.c + 1)};
}

[[gnu::always_inline]] Neighbours corners(Pt p){
  return s::array<Pt, 4>{Pt(p.r - 1, p.c - 1), Pt(p.r + 1, p.c - 1), Pt(p.r - 1, p.c + 1), Pt(p.r + 1, p.c + 1)};
}

enum class M : ubyte {
  Play    = 0x1,
  Pass    = 0x2,
  Resign  = 0x3,
  Unknown = 0x0,
};

struct Move {
  M  mty;
  Pt mpt;

  explicit Move(M ty): mty(ty) {
    assert(mty != M::Play);
  }
  Move(M ty, Pt p): mty(ty), mpt(p) {}
  Move(const Move& o): mty(o.mty), mpt(o.mpt) {}
  Move& operator=(const Move& o){
    mty = o.mty;
    mpt = o.mpt;
    return *this;
  }
};

bool operator==(const Move& a, const Move& b){
  switch (a.mty){
  case M::Play:
    return a.mty == b.mty && a.mpt == b.mpt;
  case M::Pass:
  case M::Resign:
  case M::Unknown:
    return a.mty == b.mty;
  }
}

bool operator!=(const Move& a, const Move& b){
  return not operator==(a, b);
}

char index_to_char(ubyte index){
  switch (index){
  case 0:  return 'A';
  case 1:  return 'B';
  case 2:  return 'C';
  case 3:  return 'D';
  case 4:  return 'E';
  case 5:  return 'F';
  case 6:  return 'G';
  case 7:  return 'H';
  case 8:  return 'J';
  case 9:  return 'K';
  case 10: return 'L';
  case 11: return 'M';
  case 12: return 'N';
  case 13: return 'O';
  case 14: return 'P';
  case 15: return 'Q';
  case 16: return 'R';
  case 17: return 'S';
  case 18: return 'T';
  default: assert(false);
  }
}

s::ostream& operator<<(s::ostream& out, Move m){
  switch (m.mty){
  case M::Play:   out << index_to_char(m.mpt.c) << (int)(m.mpt.r + 1); break;
  case M::Pass:   out << "pass"; break;
  case M::Resign: out << "resigns"; break;
  case M::Unknown:
  default: assert(false);
  }
  return out;
}

Move string_to_move(const s::string& str){
  if (str.size() < 2) return Move(M::Unknown);
  char colstr = str[0];
  s::string rowstr = str.substr(1);
  int colidx, rowidx;
  if (s::isupper(colstr)){
    if (colstr == 'I') return Move(M::Unknown);
    colidx = colstr - 'A';
    if (colstr >= 'J')
      colidx -= 1;
  } else {
    if (colstr == 'i') return Move(M::Unknown);
    colidx = colstr - 'a';
    if (colstr >= 'j')
      colidx -= 1;
  }
  try {
    rowidx = s::stoi(rowstr) - 1;
  } catch (s::invalid_argument& err){
    return Move(M::Unknown);
  }
  return Move(M::Play, Pt(rowidx, colidx));
}

s::istream& operator>>(s::istream& in, Move& m){
  s::string s; in >> s;
  m = string_to_move(s);
  return in;
}

Move string_to_move(const s::string& str, uint board_size){
  Move m = string_to_move(str);
  if (m.mty == M::Play){
    if (m.mpt.r < board_size && m.mpt.c < board_size)
      return m;
    else
      return Move(M::Unknown);
  } else
    return m;
}

enum class Player : ubyte {
  Black = 0x1,
  White = 0x2,
  Unknown = 0x3,
};

[[gnu::always_inline]] Player other_player(Player player){
  return (Player)((ubyte)player ^ 0x3U);
}

s::ostream& operator<<(s::ostream& out, Player player){
  switch (player){
  case Player::Black: out << "Black"; break;
  case Player::White: out << "White"; break;
  case Player::Unknown:
  default: assert(false);
  }
  return out;
}

struct PlayerMove {
  Player player;
  Move   move;

  PlayerMove(): player(Player::Unknown), move(M::Unknown) {}
  PlayerMove(Player p, Move m): player(p), move(m) {}
  PlayerMove(const PlayerMove& o): player(o.player), move(o.move) {}
};

bool operator==(const PlayerMove& a, const PlayerMove& b){
  return a.player == b.player && a.move == b.move;
}

bool operator!=(const PlayerMove& a, const PlayerMove& b){
  return not operator==(a, b);
}

} // rlgames

#endif//RLGAMES_TYPES
