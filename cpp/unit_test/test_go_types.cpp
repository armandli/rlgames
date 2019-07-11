#include <gtest/gtest.h>

#include <type_alias.h>
#include <types.h>
#include <go_types.h>
#include <zobrist_hash.h>

namespace R = rlgames;
namespace s = std;

static constexpr ubyte Size = 9;
static constexpr udyte ASize = 81;

struct TestGoStr : ::testing::Test {
  TestGoStr(){}
  ~TestGoStr(){}

  R::GoStr<Size> string;
};

TEST_F(TestGoStr, TestNumLiberties1){
  EXPECT_EQ(0, string.num_liberties());
}

TEST_F(TestGoStr, TestAddLiberty1){
  string.add_liberty(R::point<Size>(80));
  string.add_liberty(R::point<Size>(1));
  string.add_liberty(R::point<Size>(19));
  EXPECT_EQ(3, string.num_liberties());
}

TEST_F(TestGoStr, TestRemoveLiberty1){
  string.add_liberty(R::point<Size>(80));
  string.add_liberty(R::point<Size>(1));
  string.add_liberty(R::point<Size>(19));
  string.remove_liberty(R::point<Size>(19));
  string.remove_liberty(R::point<Size>(18));
  EXPECT_EQ(2, string.num_liberties());
}

TEST_F(TestGoStr, TestMerge1){
  s::bitset<ASize> stones1;
  s::bitset<ASize> stones2;
  s::bitset<ASize> liberties1;
  s::bitset<ASize> liberties2;

  stones1.set(10);
  stones1.set(20);
  stones1.set(30);

  stones2.set(11);
  stones2.set(12);
  stones2.set(13);

  liberties1.set(9);
  liberties1.set(11);
  liberties1.set(19);
  liberties1.set(21);
  liberties1.set(29);
  liberties1.set(31);

  liberties2.set(10);
  liberties2.set(14);

  R::GoStr<Size> string1(stones1, liberties1, R::Player::Black);
  R::GoStr<Size> string2(stones2, liberties2, R::Player::Black);

  string1.merge(string2);

  EXPECT_EQ(6, string1.num_liberties());
}

TEST_F(TestGoStr, TestMerge2){
  s::bitset<ASize> stones1;
  s::bitset<ASize> stones2;
  s::bitset<ASize> liberties1;
  s::bitset<ASize> liberties2;

  stones1.set(10);
  stones1.set(9);

  stones2.set(11);
  stones2.set(12);
  stones2.set(13);

  liberties1.set(11);
  liberties1.set(1);
  liberties1.set(19);
  liberties1.set(0);
  liberties1.set(18);

  liberties2.set(10);
  liberties2.set(2);
  liberties2.set(20);
  liberties2.set(3);
  liberties2.set(21);
  liberties2.set(4);
  liberties2.set(22);
  liberties2.set(14);

  R::GoStr<Size> string1(stones1, liberties1, R::Player::White);
  R::GoStr<Size> string2(stones2, liberties2, R::Player::White);

  string2.merge(string1);

  EXPECT_EQ(11, string2.num_liberties());
}

TEST_F(TestGoStr, TestEqual1){
  s::bitset<ASize> stones1;
  s::bitset<ASize> stones2;
  s::bitset<ASize> liberties1;
  s::bitset<ASize> liberties2;

  stones1.set(10);
  stones1.set(9);
  stones1.set(11);

  stones2.set(11);
  stones2.set(12);
  stones2.set(13);
  stones2.set(2);
  stones2.set(20);
  stones2.set(20);

  liberties1.set(11);
  liberties1.set(1);
  liberties1.set(19);
  liberties1.set(0);
  liberties1.set(18);

  liberties2.set(10);
  liberties2.set(2);
  liberties2.set(20);
  liberties2.set(3);
  liberties2.set(21);
  liberties2.set(4);
  liberties2.set(22);
  liberties2.set(14);

  R::GoStr<Size> string1(stones1, liberties1, R::Player::White);
  R::GoStr<Size> string2(stones2, liberties2, R::Player::White);

  EXPECT_FALSE(string1 == string2);
}

TEST_F(TestGoStr, TestEqual2){
  s::bitset<ASize> stones1;
  s::bitset<ASize> stones2;
  s::bitset<ASize> liberties1;
  s::bitset<ASize> liberties2;

  stones1.set(10);
  stones1.set(9);
  stones1.set(11);

  stones2.set(10);
  stones2.set(9);
  stones2.set(11);

  liberties1.set(11);
  liberties1.set(1);
  liberties1.set(19);
  liberties1.set(0);
  liberties1.set(18);

  liberties2.set(11);
  liberties2.set(1);
  liberties2.set(19);
  liberties2.set(0);
  liberties2.set(18);

  R::GoStr<Size> string1(stones1, liberties1, R::Player::White);
  R::GoStr<Size> string2(stones2, liberties2, R::Player::White);

  EXPECT_TRUE(string1 == string2);
}

TEST_F(TestGoStr, TestNotEqual1){
  s::bitset<ASize> stones1;
  s::bitset<ASize> stones2;
  s::bitset<ASize> liberties1;
  s::bitset<ASize> liberties2;

  stones1.set(10);
  stones1.set(9);
  stones1.set(11);

  stones2.set(10);
  stones2.set(9);
  stones2.set(11);

  liberties1.set(11);
  liberties1.set(1);
  liberties1.set(19);
  liberties1.set(0);
  liberties1.set(18);

  liberties2.set(11);
  liberties2.set(1);
  liberties2.set(19);
  liberties2.set(0);
  liberties2.set(18);

  R::GoStr<Size> string1(stones1, liberties1, R::Player::White);
  R::GoStr<Size> string2(stones2, liberties2, R::Player::White);

  EXPECT_FALSE(string1 != string2);
}

struct MockGoBoard : public R::GoBoard<Size> {
  using R::GoBoard<Size>::get_string_idx;
  using R::GoBoard<Size>::replace_string;
  using R::GoBoard<Size>::remove_string;

  MockGoBoard(): GoBoard(){}
  MockGoBoard(const MockGoBoard& o): GoBoard(o) {}
  MockGoBoard& operator=(const MockGoBoard& o){
    R::GoBoard<Size>& ret = static_cast<R::GoBoard<Size>*>(this)->operator=(static_cast<const R::GoBoard<Size>&>(o));
    return static_cast<MockGoBoard&>(ret);
  }
};

struct TestGoBoard : ::testing::Test {
  TestGoBoard(){}
  ~TestGoBoard(){}

  MockGoBoard board;
};

TEST_F(TestGoBoard, TestSize1){
  EXPECT_EQ(9, board.size());
}

TEST_F(TestGoBoard, TestHash1){
  EXPECT_EQ(0U, board.hash());
}

TEST_F(TestGoBoard, TestIsOnGrid1){
  EXPECT_TRUE(board.is_on_grid(R::Pt(4, 4)));
}

TEST_F(TestGoBoard, TestIsOnGrid2){
  EXPECT_FALSE(board.is_on_grid(R::Pt(9, 4)));
}

TEST_F(TestGoBoard, TestIsOnGrid3){
  EXPECT_FALSE(board.is_on_grid(R::Pt(4, 9)));
}

TEST_F(TestGoBoard, TestIsOnGrid4){
  EXPECT_TRUE(board.is_on_grid(R::Pt(0, 0)));
}

TEST_F(TestGoBoard, TestIsOnGrid5){
  EXPECT_TRUE(board.is_on_grid(R::Pt(8, 8)));
}

TEST_F(TestGoBoard, TestGet1){
  EXPECT_EQ(R::Player::Unknown, board.get(R::Pt(3,3)));
}

TEST_F(TestGoBoard, TestPlaceStone1){
  board.place_stone(R::Player::Black, R::Pt(1, 1));
  EXPECT_EQ(R::Player::Black, board.get(R::Pt(1, 1)));

  uint expected = R::zobrist_hash<Size>(R::Player::Black, R::Pt(1, 1));

  EXPECT_EQ(expected, board.hash());
  EXPECT_TRUE(board.get_string(R::Pt(1, 1)) != nullptr);
  EXPECT_TRUE(board.get_string(R::Pt(4, 4)) == nullptr);

  uint idx = board.get_string_idx(R::Pt(1, 1));
  EXPECT_EQ(0, idx);
}

TEST_F(TestGoBoard, TestPlaceStone2){
  board.place_stone(R::Player::White, R::Pt(7, 8));
  EXPECT_EQ(R::Player::White, board.get(R::Pt(7, 8)));

  uint expected = R::zobrist_hash<Size>(R::Player::White, R::Pt(7, 8));

  EXPECT_EQ(expected, board.hash());
  EXPECT_TRUE(board.get_string(R::Pt(7, 8)) != nullptr);
  EXPECT_TRUE(board.get_string(R::Pt(2, 8)) == nullptr);
}

TEST_F(TestGoBoard, TestPlaceStone3){
  board.place_stone(R::Player::Black, R::Pt(2, 3));
  board.place_stone(R::Player::White, R::Pt(2, 4));
  board.place_stone(R::Player::Black, R::Pt(3, 2));
  board.place_stone(R::Player::White, R::Pt(3, 3));
  board.place_stone(R::Player::Black, R::Pt(4, 3));
  board.place_stone(R::Player::White, R::Pt(4, 4));
  board.place_stone(R::Player::Black, R::Pt(3, 4));

  EXPECT_TRUE(board.get_string(R::Pt(3, 3)) == nullptr);
  EXPECT_TRUE(board.get(R::Pt(3, 3)) == R::Player::Unknown);
  EXPECT_TRUE(board.get_string(R::Pt(3, 4)) != nullptr);
  EXPECT_TRUE(board.get(R::Pt(3, 4)) == R::Player::Black);
}

TEST_F(TestGoBoard, TestPlaceStone4){
  board.place_stone(R::Player::Black, R::Pt(2, 3));
  board.place_stone(R::Player::White, R::Pt(2, 4));
  board.place_stone(R::Player::Black, R::Pt(3, 2));
  board.place_stone(R::Player::White, R::Pt(3, 3));
  board.place_stone(R::Player::Black, R::Pt(4, 3));
  board.place_stone(R::Player::White, R::Pt(4, 4));
  board.place_stone(R::Player::Black, R::Pt(3, 4));
  board.place_stone(R::Player::White, R::Pt(3, 5));

  EXPECT_TRUE(board.get_string(R::Pt(3, 4)) != nullptr);
  EXPECT_EQ(R::Player::Black, board.get(R::Pt(3, 4)));
  EXPECT_TRUE(board.get_string(R::Pt(3, 5)) != nullptr);
  EXPECT_EQ(R::Player::White, board.get(R::Pt(3, 5)));
  EXPECT_TRUE(board.get_string(R::Pt(3, 3)) == nullptr);
  EXPECT_EQ(R::Player::Unknown, board.get(R::Pt(3, 3)));
}

TEST_F(TestGoBoard, TestPlaceStone5){
  board.place_stone(R::Player::Black, R::Pt(0, 4));
  board.place_stone(R::Player::Black, R::Pt(0, 5));
  board.place_stone(R::Player::Black, R::Pt(0, 7));
  board.place_stone(R::Player::Black, R::Pt(1, 5));
  board.place_stone(R::Player::Black, R::Pt(1, 6));
  board.place_stone(R::Player::Black, R::Pt(1, 7));
  board.place_stone(R::Player::Black, R::Pt(2, 0));
  board.place_stone(R::Player::Black, R::Pt(2, 6));
  board.place_stone(R::Player::Black, R::Pt(3, 0));
  board.place_stone(R::Player::Black, R::Pt(3, 1));
  board.place_stone(R::Player::Black, R::Pt(3, 2));
  board.place_stone(R::Player::Black, R::Pt(3, 3));
  board.place_stone(R::Player::Black, R::Pt(3, 5));
  board.place_stone(R::Player::Black, R::Pt(3, 6));
  board.place_stone(R::Player::Black, R::Pt(4, 3));
  board.place_stone(R::Player::Black, R::Pt(4, 4));
  board.place_stone(R::Player::Black, R::Pt(4, 6));
  board.place_stone(R::Player::Black, R::Pt(5, 1));
  board.place_stone(R::Player::Black, R::Pt(5, 2));
  board.place_stone(R::Player::Black, R::Pt(5, 4));
  board.place_stone(R::Player::Black, R::Pt(5, 5));
  board.place_stone(R::Player::Black, R::Pt(5, 7));
  board.place_stone(R::Player::Black, R::Pt(6, 2));
  board.place_stone(R::Player::Black, R::Pt(6, 3));
  board.place_stone(R::Player::Black, R::Pt(6, 4));
  board.place_stone(R::Player::Black, R::Pt(6, 7));
  board.place_stone(R::Player::Black, R::Pt(7, 1));
  board.place_stone(R::Player::Black, R::Pt(7, 2));
  board.place_stone(R::Player::Black, R::Pt(8, 2));

  board.place_stone(R::Player::White, R::Pt(0, 3));
  board.place_stone(R::Player::White, R::Pt(0, 8));
  board.place_stone(R::Player::White, R::Pt(1, 0));
  board.place_stone(R::Player::White, R::Pt(1, 2));
  board.place_stone(R::Player::White, R::Pt(1, 4));
  board.place_stone(R::Player::White, R::Pt(1, 8));
  board.place_stone(R::Player::White, R::Pt(2, 1));
  board.place_stone(R::Player::White, R::Pt(2, 2));
  board.place_stone(R::Player::White, R::Pt(2, 3));
  board.place_stone(R::Player::White, R::Pt(2, 4));
  board.place_stone(R::Player::White, R::Pt(2, 5));
  board.place_stone(R::Player::White, R::Pt(2, 7));
  board.place_stone(R::Player::White, R::Pt(2, 8));
  board.place_stone(R::Player::White, R::Pt(3, 4));
  board.place_stone(R::Player::White, R::Pt(3, 7));
  board.place_stone(R::Player::White, R::Pt(4, 1));
  board.place_stone(R::Player::White, R::Pt(4, 7));
  board.place_stone(R::Player::White, R::Pt(5, 6));
  board.place_stone(R::Player::White, R::Pt(6, 5));
  board.place_stone(R::Player::White, R::Pt(6, 6));
  board.place_stone(R::Player::White, R::Pt(6, 8));
  board.place_stone(R::Player::White, R::Pt(7, 3));
  board.place_stone(R::Player::White, R::Pt(7, 4));
  board.place_stone(R::Player::White, R::Pt(7, 5));
  board.place_stone(R::Player::White, R::Pt(7, 7));
  board.place_stone(R::Player::White, R::Pt(8, 1));
  board.place_stone(R::Player::White, R::Pt(8, 3));

  uint idx1 = board.get_string_idx(R::Pt(5, 7));
  uint idx2 = board.get_string_idx(R::Pt(6, 7));
  EXPECT_EQ(idx1, idx2);

  idx1 = board.get_string_idx(R::Pt(2, 0));
  idx2 = board.get_string_idx(R::Pt(8, 2));
  EXPECT_EQ(idx1, idx2);

  idx1 = board.get_string_idx(R::Pt(5, 6));
  idx2 = board.get_string_idx(R::Pt(8, 3));
  EXPECT_EQ(idx1, idx2);

  idx1 = board.get_string_idx(R::Pt(0, 4));
  idx2 = board.get_string_idx(R::Pt(4, 6));
  EXPECT_EQ(idx1, idx2);
}

TEST_F(TestGoBoard, TestPlaceStone6){
  board.place_stone(R::Player::Black, R::Pt(0, 4));
  board.place_stone(R::Player::Black, R::Pt(0, 5));
  board.place_stone(R::Player::Black, R::Pt(0, 7));
  board.place_stone(R::Player::Black, R::Pt(1, 5));

  uint idx1 = board.get_string_idx(R::Pt(0, 4));
  uint idx2 = board.get_string_idx(R::Pt(0, 5));
  EXPECT_EQ(idx1, idx2);

  uint idx3 = board.get_string_idx(R::Pt(0, 7));
  EXPECT_FALSE(idx1 == idx3);
}

TEST_F(TestGoBoard, TestGetString1){
  board.place_stone(R::Player::Black, R::Pt(2, 3));
  board.place_stone(R::Player::White, R::Pt(2, 4));
  board.place_stone(R::Player::Black, R::Pt(3, 2));
  board.place_stone(R::Player::White, R::Pt(3, 3));
  board.place_stone(R::Player::Black, R::Pt(4, 3));
  board.place_stone(R::Player::White, R::Pt(4, 4));
  board.place_stone(R::Player::Black, R::Pt(3, 4));
  board.place_stone(R::Player::White, R::Pt(2, 5));

  ASSERT_TRUE(board.get_string(R::Pt(2, 5)) != nullptr);

  const R::GoStr<Size>* string = board.get_string(R::Pt(2, 5));
  EXPECT_EQ(4, string->num_liberties());
}

TEST_F(TestGoBoard, TestGetString2){
  board.place_stone(R::Player::Black, R::Pt(2, 3));
  board.place_stone(R::Player::White, R::Pt(2, 4));
  board.place_stone(R::Player::Black, R::Pt(3, 2));
  board.place_stone(R::Player::White, R::Pt(3, 3));
  board.place_stone(R::Player::Black, R::Pt(4, 3));
  board.place_stone(R::Player::White, R::Pt(4, 4));
  board.place_stone(R::Player::Black, R::Pt(3, 4));
  board.place_stone(R::Player::White, R::Pt(2, 5));

  EXPECT_TRUE(board.get_string(R::Pt(3, 5)) == nullptr);
}

TEST_F(TestGoBoard, TestRemoveString1){
  board.place_stone(R::Player::Black, R::Pt(2, 3));
  board.place_stone(R::Player::White, R::Pt(2, 4));
  board.place_stone(R::Player::Black, R::Pt(3, 2));
  board.place_stone(R::Player::White, R::Pt(3, 3));
  board.place_stone(R::Player::Black, R::Pt(4, 3));
  board.place_stone(R::Player::White, R::Pt(4, 4));
  board.place_stone(R::Player::Black, R::Pt(3, 4));
  board.place_stone(R::Player::White, R::Pt(2, 5));

  ASSERT_TRUE(board.get_string(R::Pt(2, 5)) != nullptr);
  const R::GoStr<Size>* string = board.get_string(R::Pt(2, 5));
  udyte index = board.get_string_idx(R::Pt(2, 5));
  board.remove_string(*string, index);
  EXPECT_TRUE(board.get_string(R::Pt(2, 4)) == nullptr);
  EXPECT_TRUE(board.get_string(R::Pt(2, 5)) == nullptr);
  EXPECT_TRUE(board.get_string(R::Pt(3, 2)) != nullptr);
  EXPECT_TRUE(board.get_string(R::Pt(3, 4)) != nullptr);
}

struct MockGoGameState : public R::GoGameState<Size> {
  MockGoGameState(): GoGameState(){}
  MockGoGameState(const R::GoBoard<Size>& board, R::Player player, R::Move pm, R::Move ppm, const s::unordered_set<uint>& history):
    GoGameState(board, player, pm, ppm, history){}
  MockGoGameState(const MockGoGameState& o): GoGameState(static_cast<const GoGameState&>(o)){}
  MockGoGameState& operator=(const MockGoGameState& o){
    GoGameState<Size>& ret = static_cast<GoGameState<Size>*>(this)->operator=(static_cast<const GoGameState<Size>&>(o));
    return static_cast<MockGoGameState&>(ret);
  }

  using GoGameState::is_move_self_capture;
  using GoGameState::does_move_violate_ko;
};

struct TestGoGameState : ::testing::Test {
  TestGoGameState(){}
  ~TestGoGameState(){}

  MockGoGameState state;
};

TEST_F(TestGoGameState, IsOver1){
  EXPECT_EQ(false, state.is_over());
}

TEST_F(TestGoGameState, IsOver2){
  state.apply_move(R::Move(R::M::Resign));
  EXPECT_EQ(true, state.is_over());
}

TEST_F(TestGoGameState, TestIsMoveSelfCapture1){
  state.apply_move(R::Move(R::M::Play, R::Pt(2, 3)));
  state.apply_move(R::Move(R::M::Play, R::Pt(2, 4)));
  state.apply_move(R::Move(R::M::Play, R::Pt(3, 2)));
  state.apply_move(R::Move(R::M::Play, R::Pt(3, 3)));
  state.apply_move(R::Move(R::M::Play, R::Pt(4, 3)));
  state.apply_move(R::Move(R::M::Play, R::Pt(4, 4)));
  state.apply_move(R::Move(R::M::Play, R::Pt(3, 4)));

  EXPECT_TRUE(state.is_move_self_capture(R::Move(R::M::Play, R::Pt(3, 3))));
}

TEST_F(TestGoGameState, TestDoesMoveViolateKo1){
  state.apply_move(R::Move(R::M::Play, R::Pt(2, 3)));
  state.apply_move(R::Move(R::M::Play, R::Pt(2, 4)));
  state.apply_move(R::Move(R::M::Play, R::Pt(3, 2)));
  state.apply_move(R::Move(R::M::Play, R::Pt(3, 5)));
  state.apply_move(R::Move(R::M::Play, R::Pt(4, 3)));
  state.apply_move(R::Move(R::M::Play, R::Pt(4, 4)));
  state.apply_move(R::Move(R::M::Play, R::Pt(3, 4)));
  state.apply_move(R::Move(R::M::Play, R::Pt(3, 3)));

  EXPECT_TRUE(state.does_move_violate_ko(R::Move(R::M::Play, R::Pt(3, 4))));
}

TEST_F(TestGoGameState, TestValidMove1){
  state.apply_move(R::Move(R::M::Resign));

  EXPECT_FALSE(state.is_valid_move(R::Move(R::M::Play, R::Pt(1, 1))));
}

TEST_F(TestGoGameState, TestValidMove2){
  EXPECT_TRUE(state.is_valid_move(R::Move(R::M::Pass)));
}

TEST_F(TestGoGameState, TestValidMove3){
  EXPECT_TRUE(state.is_valid_move(R::Move(R::M::Resign)));
}

TEST_F(TestGoGameState, TestValidMove4){
  state.apply_move(R::Move(R::M::Play, R::Pt(6, 6)));
  EXPECT_FALSE(state.is_valid_move(R::Move(R::M::Play, R::Pt(6, 6))));
}

TEST_F(TestGoGameState, TestValidMove5){
  state.apply_move(R::Move(R::M::Play, R::Pt(6, 6)));
  state.apply_move(R::Move(R::M::Play, R::Pt(2, 6)));
  EXPECT_FALSE(state.is_valid_move(R::Move(R::M::Play, R::Pt(2, 6))));
}

TEST_F(TestGoGameState, TestLegalMoves1){
  s::vector<R::Move> ret = state.legal_moves();

  EXPECT_EQ(ASize + 2, ret.size());
}

TEST_F(TestGoGameState, TestLegalMoves2){
  state.apply_move(R::Move(R::M::Play, R::Pt(1, 1)));
  s::vector<R::Move> ret = state.legal_moves();

  EXPECT_EQ(ASize + 1, ret.size());
}

TEST_F(TestGoGameState, TestLegalMoves3){
  state.apply_move(R::Move(R::M::Play, R::Pt(5, 3)));
  state.apply_move(R::Move(R::M::Play, R::Pt(5, 4)));
  state.apply_move(R::Move(R::M::Play, R::Pt(6, 2)));
  state.apply_move(R::Move(R::M::Play, R::Pt(6, 5)));
  state.apply_move(R::Move(R::M::Play, R::Pt(7, 3)));
  state.apply_move(R::Move(R::M::Play, R::Pt(7, 4)));
  state.apply_move(R::Move(R::M::Play, R::Pt(6, 4)));
  state.apply_move(R::Move(R::M::Play, R::Pt(6, 3)));

  s::vector<R::Move> ret = state.legal_moves();

  EXPECT_EQ(ASize - 6, ret.size());
}

TEST_F(TestGoGameState, TestWinner1){
  state.apply_move(R::Move(R::M::Resign));

  EXPECT_EQ(R::Player::White, state.winner());
}

struct MockGoAreaScore : public R::GoAreaScore<Size> {
  MockGoAreaScore(MockGoBoard& board, float komi = 7.5):
    R::GoAreaScore<Size>(static_cast<R::GoBoard<Size>&>(board), komi) {}
  using R::GoAreaScore<Size>::create_territory_labeling;
  using R::GoAreaScore<Size>::compute_score;
};

struct TestGoAreaScore : ::testing::Test {
  TestGoAreaScore(): board(), scorer(board){
    board.place_stone(R::Player::Black, R::Pt(0, 4));
    board.place_stone(R::Player::Black, R::Pt(0, 5));
    board.place_stone(R::Player::Black, R::Pt(0, 7));
    board.place_stone(R::Player::Black, R::Pt(1, 5));
    board.place_stone(R::Player::Black, R::Pt(1, 6));
    board.place_stone(R::Player::Black, R::Pt(1, 7));
    board.place_stone(R::Player::Black, R::Pt(2, 0));
    board.place_stone(R::Player::Black, R::Pt(2, 6));
    board.place_stone(R::Player::Black, R::Pt(3, 0));
    board.place_stone(R::Player::Black, R::Pt(3, 1));
    board.place_stone(R::Player::Black, R::Pt(3, 2));
    board.place_stone(R::Player::Black, R::Pt(3, 3));
    board.place_stone(R::Player::Black, R::Pt(3, 5));
    board.place_stone(R::Player::Black, R::Pt(3, 6));
    board.place_stone(R::Player::Black, R::Pt(4, 3));
    board.place_stone(R::Player::Black, R::Pt(4, 4));
    board.place_stone(R::Player::Black, R::Pt(4, 6));
    board.place_stone(R::Player::Black, R::Pt(5, 1));
    board.place_stone(R::Player::Black, R::Pt(5, 2));
    board.place_stone(R::Player::Black, R::Pt(5, 4));
    board.place_stone(R::Player::Black, R::Pt(5, 5));
    board.place_stone(R::Player::Black, R::Pt(5, 7));
    board.place_stone(R::Player::Black, R::Pt(6, 2));
    board.place_stone(R::Player::Black, R::Pt(6, 3));
    board.place_stone(R::Player::Black, R::Pt(6, 4));
    board.place_stone(R::Player::Black, R::Pt(6, 7));
    board.place_stone(R::Player::Black, R::Pt(7, 1));
    board.place_stone(R::Player::Black, R::Pt(7, 2));
    board.place_stone(R::Player::Black, R::Pt(8, 2));

    board.place_stone(R::Player::White, R::Pt(0, 3));
    board.place_stone(R::Player::White, R::Pt(0, 8));
    board.place_stone(R::Player::White, R::Pt(1, 0));
    board.place_stone(R::Player::White, R::Pt(1, 2));
    board.place_stone(R::Player::White, R::Pt(1, 4));
    board.place_stone(R::Player::White, R::Pt(1, 8));
    board.place_stone(R::Player::White, R::Pt(2, 1));
    board.place_stone(R::Player::White, R::Pt(2, 2));
    board.place_stone(R::Player::White, R::Pt(2, 3));
    board.place_stone(R::Player::White, R::Pt(2, 4));
    board.place_stone(R::Player::White, R::Pt(2, 5));
    board.place_stone(R::Player::White, R::Pt(2, 7));
    board.place_stone(R::Player::White, R::Pt(2, 8));
    board.place_stone(R::Player::White, R::Pt(3, 4));
    board.place_stone(R::Player::White, R::Pt(3, 7));
    board.place_stone(R::Player::White, R::Pt(4, 1));
    board.place_stone(R::Player::White, R::Pt(4, 7));
    board.place_stone(R::Player::White, R::Pt(5, 6));
    board.place_stone(R::Player::White, R::Pt(6, 5));
    board.place_stone(R::Player::White, R::Pt(6, 6));
    board.place_stone(R::Player::White, R::Pt(6, 8));
    board.place_stone(R::Player::White, R::Pt(7, 3));
    board.place_stone(R::Player::White, R::Pt(7, 4));
    board.place_stone(R::Player::White, R::Pt(7, 5));
    board.place_stone(R::Player::White, R::Pt(7, 7));
    board.place_stone(R::Player::White, R::Pt(8, 1));
    board.place_stone(R::Player::White, R::Pt(8, 3));
  }
  ~TestGoAreaScore(){}

  MockGoBoard     board;
  MockGoAreaScore scorer;
};

static constexpr ubyte white = (ubyte)R::Player::White;
static constexpr ubyte black = (ubyte)R::Player::Black;
static constexpr ubyte dame  = white | black;

TEST_F(TestGoAreaScore, TestTerritoryLabeling1){
  EXPECT_EQ(R::Player::White, board.get(R::Pt(4, 1)));
  EXPECT_EQ(R::Player::White, board.get(R::Pt(8, 1)));
  EXPECT_EQ(R::Player::Black, board.get(R::Pt(5, 7)));
  EXPECT_EQ(R::Player::Black, board.get(R::Pt(6, 7)));

  s::array<ubyte, ASize> ret = scorer.create_territory_labeling();
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(0, 0))]);
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(0, 1))]);
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(0, 2))]);
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(1, 1))]);
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(8, 4))]);
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(8, 5))]);
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(8, 6))]);
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(7, 6))]);
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(8, 7))]);
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(8, 8))]);
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(7, 8))]);
  EXPECT_EQ(white, ret[R::index<Size>(R::Pt(1, 3))]);

  EXPECT_EQ(black, ret[R::index<Size>(R::Pt(0, 6))]);
  EXPECT_EQ(black, ret[R::index<Size>(R::Pt(4, 5))]);
  EXPECT_EQ(black, ret[R::index<Size>(R::Pt(5, 3))]);

  EXPECT_EQ(dame, ret[R::index<Size>(R::Pt(3, 8))]);
  EXPECT_EQ(dame, ret[R::index<Size>(R::Pt(4, 8))]);
  EXPECT_EQ(dame, ret[R::index<Size>(R::Pt(5, 8))]);
  EXPECT_EQ(dame, ret[R::index<Size>(R::Pt(4, 0))]);
  EXPECT_EQ(dame, ret[R::index<Size>(R::Pt(5, 0))]);
  EXPECT_EQ(dame, ret[R::index<Size>(R::Pt(6, 0))]);
  EXPECT_EQ(dame, ret[R::index<Size>(R::Pt(7, 0))]);
  EXPECT_EQ(dame, ret[R::index<Size>(R::Pt(8, 0))]);
  EXPECT_EQ(dame, ret[R::index<Size>(R::Pt(4, 2))]);
  EXPECT_EQ(dame, ret[R::index<Size>(R::Pt(6, 1))]);

  EXPECT_EQ(0U, ret[R::index<Size>(R::Pt(0, 4))]);
  EXPECT_EQ(0U, ret[R::index<Size>(R::Pt(4, 1))]);
}

TEST_F(TestGoAreaScore, TestWinner1){
  EXPECT_EQ(R::Player::White, scorer.winner());
}

TEST_F(TestGoAreaScore, TestWinningMargin1){
  EXPECT_EQ(-14.5, scorer.winning_margin());
}

//TODO: test zobrist hash being correct after string removal
