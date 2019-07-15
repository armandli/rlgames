#include <gtest/gtest.h>

#include <type_alias.h>
#include <types.h>
#include <ttt_types.h>

namespace R = rlgames;
namespace s = std;

struct TestBoard : ::testing::Test {
  TestBoard(){}
  ~TestBoard(){}

  R::TTTBoard board;
};

TEST_F(TestBoard, TestIsOnGrid1){
  R::Pt p(2, 2);
  EXPECT_TRUE(board.is_on_grid(p));
}

TEST_F(TestBoard, TestIsOnGrid2){
  R::Pt p(10, 1);
  EXPECT_FALSE(board.is_on_grid(p));
}

TEST_F(TestBoard, TestIsOnGrid3){
  R::Pt p(0, 4);
  EXPECT_FALSE(board.is_on_grid(p));
}

TEST_F(TestBoard, TestIsOnGrid4){
  R::Pt p(12, 11);
  EXPECT_FALSE(board.is_on_grid(p));
}

TEST_F(TestBoard, TestIsOnGrid5){
  R::Pt p(0xFFU, 0xFFU);
  EXPECT_FALSE(board.is_on_grid(p));
}

TEST_F(TestBoard, TestGet1){
  R::Pt p(2, 2);
  R::Player player = board.get(p);
  EXPECT_EQ(R::Player::Unknown, player);
}

TEST_F(TestBoard, TestGet2){
  R::Pt p(2, 2);
  board.place_stone(R::Player::White, p);
  R::Player player = board.get(p);
  EXPECT_EQ(R::Player::White, player);
}

TEST_F(TestBoard, TestPlaceStone1){
  for (size_t i = 0; i < 3; ++i){
    R::Pt pt(i, i);
    board.place_stone(R::Player::Black, pt);
  }
  for (size_t i = 0; i < 9; ++i){
    R::Pt pt = R::point<3>(i);
    R::Player player = board.get(pt);
    if (pt.r == pt.c)
      EXPECT_EQ(R::Player::Black, player);
    else
      EXPECT_EQ(R::Player::Unknown, player);
  }
}

TEST_F(TestBoard, TestSize1){
  EXPECT_EQ(3, board.size());
}

class TestTTTGameState : public R::TTTGameState {
public:
  TestTTTGameState() : TTTGameState() {}
  TestTTTGameState(const R::TTTBoard& board, R::Player player, R::Move move): TTTGameState(board, player, move) {}
  using TTTGameState::is_connected;
};

struct TestGameState : ::testing::Test {
  TestGameState(){}
  ~TestGameState(){}

  TestTTTGameState gs;
};

TEST_F(TestGameState, TestLegalMoves1){
  s::vector<R::Move> moves = gs.legal_moves();
  EXPECT_EQ(10, moves.size());
}

TEST_F(TestGameState, TestLegalMoves2){
  gs.apply_move(R::Move(R::M::Play, R::Pt(1, 2)));
  gs.apply_move(R::Move(R::M::Play, R::Pt(2, 1)));
  gs.apply_move(R::Move(R::M::Play, R::Pt(0, 0)));

  s::vector<R::Move> moves = gs.legal_moves();
  EXPECT_EQ(7, moves.size());
}

TEST_F(TestGameState, TestIsOver1){
  gs.apply_move(R::Move(R::M::Play, R::Pt(1, 1)));
  EXPECT_FALSE(gs.is_over());
}

TEST_F(TestGameState, TestIsOver2){
  gs.apply_move(R::Move(R::M::Play, R::Pt(1, 1)));
  gs.apply_move(R::Move(R::M::Play, R::Pt(0, 1)));
  gs.apply_move(R::Move(R::M::Play, R::Pt(1, 0)));
  gs.apply_move(R::Move(R::M::Play, R::Pt(2, 1)));
  gs.apply_move(R::Move(R::M::Play, R::Pt(1, 2)));

  EXPECT_TRUE(gs.is_over());
}

TEST_F(TestGameState, TestWinner1){
  gs.apply_move(R::Move(R::M::Play, R::Pt(1, 1)));
  gs.apply_move(R::Move(R::M::Play, R::Pt(0, 1)));
  gs.apply_move(R::Move(R::M::Play, R::Pt(1, 0)));
  gs.apply_move(R::Move(R::M::Play, R::Pt(2, 1)));
  gs.apply_move(R::Move(R::M::Play, R::Pt(1, 2)));

  EXPECT_EQ(R::Player::Black, gs.winner());
}

TEST_F(TestGameState, TestIsConnected1){
  R::TTTBoard board;
  board.place_stone(R::Player::Black, R::Pt(0, 2));
  board.place_stone(R::Player::Black, R::Pt(1, 1));
  board.place_stone(R::Player::Black, R::Pt(2, 0));
  TestTTTGameState state(board, R::Player::White, R::Move(R::M::Play, R::Pt(2, 0)));

  EXPECT_TRUE(state.is_over());
}

TEST_F(TestGameState, TestIsConnected2){
  R::TTTBoard board;
  board.place_stone(R::Player::Black, R::Pt(0, 0));
  board.place_stone(R::Player::Black, R::Pt(1, 1));
  board.place_stone(R::Player::Black, R::Pt(2, 2));
  TestTTTGameState state(board, R::Player::White, R::Move(R::M::Play, R::Pt(2, 2)));

  EXPECT_TRUE(state.is_over());
}

TEST_F(TestGameState, TestIsConnected3){
  R::TTTBoard board;
  board.place_stone(R::Player::Black, R::Pt(0, 1));
  board.place_stone(R::Player::Black, R::Pt(1, 1));
  board.place_stone(R::Player::Black, R::Pt(2, 1));
  TestTTTGameState state(board, R::Player::White, R::Move(R::M::Play, R::Pt(2, 1)));

  EXPECT_TRUE(state.is_over());
}

TEST_F(TestGameState, TestIsConnected4){
  R::TTTBoard board;
  board.place_stone(R::Player::Black, R::Pt(1, 0));
  board.place_stone(R::Player::Black, R::Pt(1, 1));
  board.place_stone(R::Player::Black, R::Pt(1, 2));
  TestTTTGameState state(board, R::Player::White, R::Move(R::M::Play, R::Pt(1, 2)));

  EXPECT_TRUE(state.is_over());
}

TEST_F(TestGameState, TestIsConnected5){
  R::TTTBoard board;
  board.place_stone(R::Player::Black, R::Pt(0, 0));
  board.place_stone(R::Player::White, R::Pt(1, 1));
  board.place_stone(R::Player::Black, R::Pt(0, 1));
  board.place_stone(R::Player::White, R::Pt(0, 2));
  board.place_stone(R::Player::Black, R::Pt(2, 0));
  board.place_stone(R::Player::White, R::Pt(1, 0));
  board.place_stone(R::Player::Black, R::Pt(1, 2));
  board.place_stone(R::Player::White, R::Pt(2, 1));
  board.place_stone(R::Player::Black, R::Pt(2, 2));
  TestTTTGameState state(board, R::Player::White, R::Move(R::M::Play, R::Pt(2, 2)));

  EXPECT_TRUE(state.is_over());
  EXPECT_EQ(R::Player::Unknown, state.winner());
}
