#include <cmath>

#include <gtest/gtest.h>

#include <type_alias.h>
#include <types.h>

namespace R = rlgames;

TEST(TestPoint, TestPtToIndex1){
  R::Pt p;
  uint index = R::index<9>(p);
  EXPECT_EQ(0U, index);
}

TEST(TestPoint, TestPtToIndex2){
  R::Pt p(2, 4);
  uint index = R::index<9>(p);
  EXPECT_EQ(22, index);
}
TEST(TestPoint, TestPtToIndex3){
  R::Pt p(8, 8);
  uint index = R::index<9>(p);
  EXPECT_EQ(80, index);
}

TEST(TestPoint, TestIdxToPt1){
  uint index = 0;
  R::Pt p = R::point<9>(index);
  EXPECT_EQ(0, p.r);
  EXPECT_EQ(0, p.c);
}

TEST(TestPoint, TestIdxToPt2){
  uint index = 16;
  R::Pt p = R::point<9>(index);
  EXPECT_EQ(1, p.r);
  EXPECT_EQ(7, p.c);
}

TEST(TestPoint, TestIdxToPt3){
  uint index = 80;
  R::Pt p = R::point<9>(index);
  EXPECT_EQ(8, p.r);
  EXPECT_EQ(8, p.c);
}

TEST(TestPoint, TestNeighbours1){
  R::Pt p(3, 7);
  R::Neighbours a = R::neighbours(p);
  ASSERT_EQ(4, a.size());

  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(1, abs(a[i].r - p.r) + abs(a[i].c - p.c));
}

TEST(TestPoint, TestNeighbours2){
  R::Pt p(0, 7);
  R::Neighbours a = R::neighbours(p);
  ASSERT_EQ(4, a.size());

  for (size_t i = 0; i < 4; ++i)
    EXPECT_LE(abs(a[i].c - p.c), 1);
}

TEST(TestPoint, TestNeighbours3){
  R::Pt p(4, 0);
  R::Neighbours a = R::neighbours(p);
  ASSERT_EQ(4, a.size());

  for (size_t i = 0; i < 4; ++i)
    EXPECT_LE(abs(a[i].r - p.r), 1);
}


TEST(TestPoint, TestCorners1){
  R::Pt p(6, 6);
  R::Neighbours a = R::corners(p);
  ASSERT_EQ(4, a.size());

  for (size_t i = 0; i < 4; ++i)
    EXPECT_EQ(2, abs(a[i].r - p.r) + abs(a[i].c - p.c));
}

TEST(TestPoint, TestCorners2){
  R::Pt p(0, 6);
  R::Neighbours a = R::corners(p);
  ASSERT_EQ(4, a.size());
}

TEST(TestPoint, TestCorners3){
  R::Pt p(6, 0);
  R::Neighbours a = R::corners(p);
  ASSERT_EQ(4, a.size());
}

TEST(TestPlayer, TestOtherPlayer){
  R::Player p   = R::Player::Black;
  R::Player oth = R::other_player(p);
  EXPECT_EQ(R::Player::White, oth);
}
