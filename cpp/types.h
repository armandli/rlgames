#ifndef RLGAMES_TYPES
#define RLGAMES_TYPES

#include <cassert>
#include <array>
#include <type_alias.h>

namespace s = std;

namespace rlgames {

struct Pt {
  ubyte r, c;

  Pt() = default;
  Pt(ubyte r, ubyte c): r(r), c(c) {}
};

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
};

enum class Player : ubyte {
  Black = 0x1,
  White = 0x2,
  Unknown = 0x3,
};

[[gnu::always_inline]] Player other_player(Player player){
  return (Player)((ubyte)player ^ 0x3U);
}

} // rlgames

#endif//RLGAMES_TYPES
