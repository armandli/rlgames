#include <gtest/gtest.h>

#include <bitset>

#include <bag.h>

namespace s = std;
namespace R = rlgames;

using bs = s::bitset<91>;

struct TestBag : ::testing::Test {
  TestBag(){}
  ~TestBag(){}

  R::bag<bs> b;
};

TEST_F(TestBag, TestCopy1){
  R::bag<bs> b2;
  b2.push_back(bs());
  b2.push_back(bs());
  b2.push_back(bs());
  b = b2;
  EXPECT_EQ(3, b.size());
}

TEST_F(TestBag, TestPush1){
  for (size_t i = 0; i < 7; ++i){
    bs obj; obj.set(i);
    b.push_back(obj);
  }
  bs obj; obj.set(7);
  b.push(3, obj);
  bs& res = b[3];
  EXPECT_TRUE(res.test(7));
}

TEST_F(TestBag, TestPop1){
  for (size_t i = 0; i < 14; ++i){
    bs obj; obj.set(i);
    b.push_back(obj);
  }
  bs obj = b.pop(10);
  EXPECT_TRUE(obj.test(10));
}

TEST_F(TestBag, TestPop2){
  bs obj; obj.set(11);
  b.push(0, obj);
  bs res = b.pop(0);
  EXPECT_TRUE(res.test(11));
}

TEST_F(TestBag, TestIterator1){
  for (size_t i = 0; i < 13; ++i){
    bs obj; obj.set(i);
    b.push_back(obj);
  }
  size_t counter = 0;
  for (bs& obj : b){
    EXPECT_TRUE(obj.test(counter++));
  }
}
