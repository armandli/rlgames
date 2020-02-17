#ifndef RLGAMES_GO_TYPES
#define RLGAMES_GO_TYPES

#include <cassert>
#include <cstring>
#include <bitset>
#include <array>
#include <vector>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <ostream>

#include <type_alias.h>
#include <types.h>
#include <bag.h>
#include <zobrist_hash.h>
#include <game_base.h>

namespace s = std;

namespace rlgames {

template <ubyte SZ>
struct GoStr {
  static constexpr uint SIZE = SZ;
  static constexpr uint IZ = SZ * SZ;
private:
  s::bitset<IZ> mStones;
  s::bitset<IZ> mLiberties;
  Player        mColor;
public:
  GoStr(): mColor(Player::Unknown){}
  GoStr(const s::bitset<IZ>& stones, const s::bitset<IZ>& liberties, Player color):
    mStones(stones), mLiberties(liberties), mColor(color) {}
  GoStr(const GoStr& o): mStones(o.mStones), mLiberties(o.mLiberties), mColor(o.mColor) {}
  GoStr& operator=(const GoStr& o){
    mStones = o.mStones;
    mLiberties = o.mLiberties;
    mColor = o.mColor;
    return *this;
  }
  GoStr(GoStr&& o) noexcept : mStones(s::move(o.mStones)), mLiberties(s::move(o.mLiberties)), mColor(o.mColor) {}
  GoStr& operator=(GoStr&& o) noexcept {
    mStones = s::move(o.mStones);
    mLiberties = s::move(o.mLiberties);
    mColor = o.mColor;
    return *this;
  }

  Player color() const { return mColor; }
  const s::bitset<IZ>& stones() const { return mStones; }
  const s::bitset<IZ>& liberties() const { return mLiberties; }
  size_t num_liberties() const { return mLiberties.count(); }

  void add_liberty(Pt pt){ mLiberties.set(index<SZ>(pt)); }
  void remove_liberty(Pt pt){ mLiberties.reset(index<SZ>(pt)); }
  void merge(const GoStr& b){
    assert(mColor == b.mColor);

    mStones   |= b.mStones;
    mLiberties = (mLiberties | b.mLiberties) & ~mStones;
  }
};

template <ubyte SZ>
[[gnu::always_inline]] bool operator==(const GoStr<SZ>& a, const GoStr<SZ>& b){
  return a.color() == b.color() && a.stones() == b.stones() && a.liberties() == b.liberties();
}

template <ubyte SZ>
[[gnu::always_inline]] bool operator!=(const GoStr<SZ>& a, const GoStr<SZ>& b){
  return not (a == b);
}

template <ubyte SZ>
[[gnu::always_inline]] GoStr<SZ> merge(const GoStr<SZ>& a, const GoStr<SZ>& b){
  assert(a.color() == b.color());

  s::bitset<SZ> stones = a.stones() | b.stones();
  s::bitset<SZ> liberties = (a.liberties() | b.liberties()) & ~stones;
  return GoStr<SZ>(stones, liberties, a.color());
}

template <ubyte SZ> struct GoAreaScore;

template <ubyte SZ>
struct GoBoard : Board<GoBoard<SZ>> {
  friend struct GoAreaScore<SZ>;
  static constexpr uint SIZE   = SZ;
  static constexpr uint IZ     = SZ * SZ;
  static constexpr udyte EMPTY = 0xFFFFU;
private:
  bag<GoStr<SZ>>      mStrings;
  s::array<udyte, IZ> mBoard;
  uint                mHash;
protected:
  udyte get_string_idx(Pt pt) const {
    assert(is_on_grid(pt));

    return mBoard[index<SZ>(pt)];
  }
  //TODO: interface redesign, does it really need both index and string ref ?
  void replace_string(const GoStr<SZ>& string, udyte index){
    const s::bitset<IZ>& stones = string.stones();
    for (uint i = 0; i < IZ; ++i)
      if (stones.test(i))
        mBoard[i] = index;
  }
  //TODO: interface redesign, does it really need both index and string ref ?
  void remove_string(const GoStr<SZ>& string, uint index){
    const s::bitset<IZ>& stones = string.stones();
    for (uint i = 0; i < IZ; ++i)
      if (stones.test(i)){
        Pt pt = point<SZ>(i);
        Neighbours ns = neighbours(pt);
        for (Pt neighbour : ns){
          if (not is_on_grid(neighbour)) continue;

          uint sidx = get_string_idx(neighbour);
          if (sidx == EMPTY) continue;
          if (sidx != index)
            mStrings[sidx].add_liberty(pt);
        }
        mBoard[i] = EMPTY;

        mHash ^= zobrist_hash<SZ>(string.color(), pt);
      }
    //TODO: seems like pop and replace_string should always go together,
    //      should place them together
    mStrings.pop(index);
    if (index < mStrings.size())
      replace_string(mStrings[index], index);
  }
public:
  GoBoard(): mHash(EMPTY_BOARD) {
    s::memset(mBoard.data(), 0xFFU, sizeof(udyte) * IZ);
  }
  GoBoard(const GoBoard& o): mStrings(o.mStrings), mBoard(o.mBoard), mHash(o.mHash) {}
  GoBoard& operator=(const GoBoard& o){
    mStrings = o.mStrings;
    mBoard   = o.mBoard;
    mHash    = o.mHash;
    return *this;
  }
  GoBoard(GoBoard&& o) noexcept : mStrings(s::move(o.mStrings)), mBoard(s::move(o.mBoard)), mHash(o.mHash) {}
  GoBoard& operator=(GoBoard&& o) noexcept {
    mStrings = s::move(o.mStrings);
    mBoard   = s::move(o.mBoard);
    mHash    = o.mHash;
    return *this;
  }

  uint size() const { return SZ; }
  uint hash() const { return mHash; }
  bool is_on_grid(Pt pt) const { return (pt.r < SZ && pt.c < SZ); }
  Player get(Pt pt) const {
    assert(is_on_grid(pt));

    uint idx = get_string_idx(pt);
    if (idx == EMPTY) return Player::Unknown;
    else              return mStrings[idx].color();
  }
  const GoStr<SZ>* get_string(Pt pt) const {
    assert(is_on_grid(pt));

    uint idx = get_string_idx(pt);
    if (idx == EMPTY) return nullptr;
    else              return &mStrings[idx];
  }
  void place_stone(Player player, Pt pt){
    assert(is_on_grid(pt));
    assert(get_string_idx(pt) == EMPTY);

    mHash ^= zobrist_hash<SZ>(player, pt);

    Player oplayer = other_player(player);
    Neighbours ns = neighbours(pt);
    s::bitset<IZ> liberties;
    s::array<udyte, 4> adj_same_color;
    s::array<udyte, 4> adj_oppo_color;
    decltype(s::begin(adj_same_color)) adj_same_iter = s::begin(adj_same_color);
    decltype(s::begin(adj_oppo_color)) adj_oppo_iter = s::begin(adj_oppo_color);

    for (Pt neighbour : ns){
      if (not is_on_grid(neighbour)) continue;

      uint sidx = get_string_idx(neighbour);
      if (sidx == EMPTY)
        liberties.set(index<SZ>(neighbour));
      else if (mStrings[sidx].color() == player)
        *(adj_same_iter++) = sidx;
    }
    s::sort(s::begin(adj_same_color), adj_same_iter, s::greater<uint>());
    adj_same_iter = s::unique(s::begin(adj_same_color), adj_same_iter);

    //merge all same color string together
    s::bitset<IZ> stones; stones.set(index<SZ>(pt));
    GoStr<SZ> new_string(stones, liberties, player);
    for (decltype(s::begin(adj_same_color)) it = s::begin(adj_same_color); it != adj_same_iter; ++it){
      new_string.merge(mStrings[*it]);
      //TODO: seems like pop and replace_string should always go together,
      //      should place them together
      mStrings.pop(*it);
      if (*it < mStrings.size())
        replace_string(mStrings[*it], *it);
    }
    udyte new_index = mStrings.push_back(new_string);
    replace_string(new_string, new_index);

    for (Pt neighbour : ns){
      if (not is_on_grid(neighbour)) continue;

      uint sidx = get_string_idx(neighbour);
      if (sidx != EMPTY && mStrings[sidx].color() == oplayer)
        *(adj_oppo_iter++) = sidx;
    }
    s::sort(s::begin(adj_oppo_color), adj_oppo_iter, s::greater<uint>());
    adj_oppo_iter = s::unique(s::begin(adj_oppo_color), adj_oppo_iter);

    //remove opponent dead string
    for (decltype(s::begin(adj_oppo_color)) it = s::begin(adj_oppo_color); it != adj_oppo_iter; ++it){
      GoStr<SZ>& string = mStrings[*it];
      string.remove_liberty(pt);
      if (string.num_liberties() == 0)
        remove_string(string, *it);
    }
    //not removing new merged string here, allow board to show self capture
    //happened for rule checking
  }

  s::ostream& print(s::ostream& out) const {
    char bchar = 'X';
    char wchar = '0';

    out << "   ";
    if (SZ > 9) out << " ";
    for (char c = 'A', i = 0; i < SZ; c++, i++){
      if (c == 'I') c++;
      out << c << ' ';
    }
    out << '\n';
    for (ubyte i = SZ - 1; i < SZ; --i){
      if (i < 10) out << ' ';
      out << i + 1 << ' ';
      for (ubyte j = 0; j < SZ; ++j){
        udyte idx = get_string_idx(Pt(i, j));
        if (idx == EMPTY)
          out << '.';
        else {
          Player color = mStrings[idx].color();
          if (color == Player::Black) out << bchar;
          else                        out << wchar;
        }
        out << ' ';
      }
      out << i + 1 << '\n';
    }
    out << "   ";
    if (SZ > 9) out << " ";
    for (char c = 'A', i = 0; i < SZ; c++, i++){
      if (c == 'I') c++;
      out << c << ' ';
    }
    out << '\n';
    return out;
  }

  s::ostream& debug(s::ostream& out) const {
    for (ubyte i = 0; i < SZ; ++i){
      for (ubyte j = 0; j < SZ; ++j){
        udyte idx = get_string_idx(Pt(i, j));
        if (idx == EMPTY)
          out << "  ";
        else {
          out << s::hex << idx << " ";
        }
      }
      out << "\n";
    }
    return out;
  }
};

template <ubyte SZ>
s::ostream& operator<<(s::ostream& out, const GoBoard<SZ>& board){
  return board.print(out);
}

template <ubyte SZ>
constexpr float default_komi(){
  if      constexpr(SZ < 5)              return 0.F;
  else if constexpr(SZ >= 5 &&  SZ < 7)  return 0.5F;
  else if constexpr(SZ >= 7 &&  SZ < 10) return 1.5F;
  else if constexpr(SZ >= 10 && SZ < 13) return 2.5F;
  else if constexpr(SZ >= 13 && SZ < 14) return 3.5F;
  else if constexpr(SZ >= 14 && SZ < 16) return 4.5F;
  else if constexpr(SZ >= 16 && SZ < 17) return 5.5F;
  else if constexpr(SZ >= 17 && SZ < 19) return 6.5F;
  else                                   return 7.5F;
}

// scoring using area rule: player pieces on board + territory + komi
template <ubyte SZ>
struct GoAreaScore {
  static constexpr uint SIZE = SZ;
  static constexpr uint IZ = SZ * SZ;
  static constexpr ubyte DAME = (ubyte) Player::White | (ubyte) Player::Black;
private:
  const GoBoard<SZ>& mBoard;
  float              mBlackPoints;
  float              mBlackTerritory;
  float              mWhitePoints;
  float              mWhiteTerritory;
  float              mDames;
  float              mKomi;
protected:
  s::array<ubyte, IZ> create_territory_labeling(){
    struct Recursion {
      Recursion(const GoBoard<SZ>& board): board(board) {}
      ubyte operator()(Pt spt, s::bitset<IZ>& points){
        points.set(index<SZ>(spt));
        ubyte coloring = 0;
        for (Pt neighbour : neighbours(spt)){
          if (not board.is_on_grid(neighbour)) continue;
          Player color = board.get(neighbour);
          if ((not points.test(index<SZ>(neighbour))) && color == Player::Unknown){
            ubyte rret = this->operator()(neighbour, points);
            coloring |= rret;
          } else if (color != Player::Unknown){
            coloring |= (ubyte)color;
          }
        }
        if (coloring == 0) return DAME;
        else               return coloring;
      }
    private:
      const GoBoard<SZ>& board;
    } recursive_labeling(mBoard);

    s::array<ubyte, IZ> labels;
    s::memset(labels.data(), 0U, sizeof(ubyte) * IZ);
    for (uint i = 0; i < IZ; ++i){
      Pt pt = point<SZ>(i);
      if (mBoard.get(pt) == Player::Unknown && labels[i] == 0U){
        s::bitset<IZ> points;
        ubyte coloring = recursive_labeling(pt, points);
        for (uint i = 0; i < IZ; ++i)
          if (points.test(i))
            labels[i] = coloring;
      }
    }
    return labels;
  }

  void compute_score(const s::array<ubyte, IZ>& labels){
    for (uint i = 0; i < IZ; ++i){
      Pt pt = point<SZ>(i);
      Player color = mBoard.get(pt);
      switch (color){
      case Player::Black: mBlackPoints++; break;
      case Player::White: mWhitePoints++; break;
      case Player::Unknown:
        switch (labels[i]){
        case (ubyte)Player::Black: mBlackTerritory++; break;
        case (ubyte)Player::White: mWhiteTerritory++; break;
        case DAME:          mDames++;          break;
        default: assert(false);
        }
        break;
      default: assert(false);
      }
    }
  }

  void compute_winner(){
    s::array<ubyte, IZ> labels = create_territory_labeling();
    compute_score(labels);
  }

  bool is_winner_computed(){
    if (mBlackPoints == 0.F && mWhitePoints == 0.F && mBlackTerritory == 0.F && mWhiteTerritory == 0.F)
      return false;
    else
      return true;
  }
public:
  explicit GoAreaScore(const GoBoard<SZ>& board, float komi = 7.5):
    mBoard(board),
    mBlackPoints(0.F), mBlackTerritory(0.F),
    mWhitePoints(0.F), mWhiteTerritory(0.F),
    mDames(0), mKomi(komi) {}
  Player winner(){
    if (not is_winner_computed()) compute_winner();
    if (mBlackPoints + mBlackTerritory > mWhitePoints + mWhiteTerritory + mKomi)
      return Player::Black;
    else
      return Player::White;
  }

  float winning_margin(){
    if (not is_winner_computed()) compute_winner();
    return mBlackPoints - mWhitePoints + mBlackTerritory - mWhiteTerritory - mKomi;
  }
};

template <ubyte SZ>
struct GoGameState : GameState<GoBoard<SZ>, GoGameState<SZ>> {
  static constexpr uint SIZE = SZ;
  static constexpr uint IZ = SZ * SZ;

protected:
  GoBoard<SZ>            mBoard;
  Player                 mNPlayer; //next player
  Move                   mPMove;   //previous move
  Move                   mPPMove;  //previous previous move
  s::unordered_set<uint> mHistory; //zobrist hash history
protected:
  //self capture is optionally allowed in Go, but we assume it is always bad
  //and prune
  bool is_move_self_capture(Move move) const {
    if (move.mty != M::Play) return false;
    GoBoard<SZ> test_board = mBoard;
    test_board.place_stone(mNPlayer, move.mpt);
    const GoStr<SZ>* string = test_board.get_string(move.mpt);
    return string->num_liberties() == 0;
  }
public:
  GoGameState():
    mBoard(), mNPlayer(Player::Black), mPMove(M::Unknown), mPPMove(M::Unknown), mHistory() {}
  GoGameState(const GoBoard<SZ>& board, Player player, Move pmove, Move ppmove, const s::unordered_set<uint>& history):
    mBoard(board), mNPlayer(player), mPMove(pmove), mPPMove(ppmove), mHistory(history) {}
  GoGameState(const GoGameState& o):
    mBoard(o.mBoard), mNPlayer(o.mNPlayer), mPMove(o.mPMove), mPPMove(o.mPPMove), mHistory(o.mHistory) {}
  GoGameState& operator=(const GoGameState& o){
    mBoard = o.mBoard;
    mNPlayer = o.mNPlayer;
    mPMove = o.mPMove;
    mPPMove = o.mPPMove;
    mHistory = o.mHistory;
    return *this;
  }
  GoGameState(GoGameState&& o) noexcept :
    mBoard(s::move(o.mBoard)),
    mNPlayer(o.mNPlayer),
    mPMove(o.mPMove),
    mPPMove(o.mPPMove),
    mHistory(s::move(o.mHistory))
  {}
  GoGameState& operator=(GoGameState&& o) noexcept {
    mBoard = s::move(o.mBoard);
    mNPlayer = o.mNPlayer;
    mPMove = o.mPMove;
    mPPMove = o.mPPMove;
    mHistory = s::move(o.mHistory);
    return *this;
  }

  const GoBoard<SZ>& board() const { return mBoard; }
  Player next_player() const { return mNPlayer; }
  Move previous_move() const { return mPMove; }

  bool is_over() const {
    if (mPMove.mty == M::Unknown)  return false;
    if (mPMove.mty == M::Resign)   return true;
    if (mPPMove.mty == M::Unknown) return false;
    return (mPMove.mty == M::Pass && mPPMove.mty == M::Pass);
  }
  bool does_move_violate_ko(Move move) const {
    if (move.mty != M::Play) return false;
    GoBoard<SZ> test_board = mBoard;
    test_board.place_stone(mNPlayer, move.mpt);
    return mHistory.find(test_board.hash()) != s::end(mHistory);
  }
  bool is_valid_move(Move move) const {
    if (is_over())                                    return false;
    if (move.mty == M::Pass || move.mty == M::Resign) return true;
    // does not allow self capture in this game
    return
      mBoard.get(move.mpt) == Player::Unknown &&
      not is_move_self_capture(move) &&
      not does_move_violate_ko(move);
  }
  s::vector<Move> legal_moves() const {
    if (is_over()) return s::vector<Move>();
    s::vector<Move> ret; ret.reserve(IZ);
    for (uint r = 0; r < SZ; ++r)
      for (uint c = 0; c < SZ; ++c){
        Move m(M::Play, Pt(r, c));
        if (is_valid_move(m))
          ret.push_back(m);
      }
    ret.push_back(Move(M::Pass));
    ret.push_back(Move(M::Resign));
    return ret;
  }
  Player winner(){
    if (not is_over()) return Player::Unknown;
    if (mPMove.mty == M::Resign) return mNPlayer;
    GoAreaScore<SZ> scorer(mBoard, default_komi<SZ>());
    return scorer.winner();
  }
  GoGameState& apply_move(Move move){
    mPPMove = mPMove;
    mPMove = move;
    if (move.mty == M::Play){
      mBoard.place_stone(mNPlayer, move.mpt);
      mHistory.insert(mBoard.hash());
    }
    mNPlayer = other_player(mNPlayer);
    return *this;
  }
};

} //rlgames

#endif//RLGAMES_GO_TYPES
