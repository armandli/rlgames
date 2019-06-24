#include <gtest/gtest.h>

#include <bitset>
#include <vector>
#include <functional>
#include <algorithm>

#include <type_alias.h>
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
  b.pop(10);
  for (size_t i = 0; i < b.size(); ++i)
    EXPECT_FALSE(b[i].test(10) == true);
}

TEST_F(TestBag, TestPop2){
  bs obj; obj.set(11);
  b.push(0, obj);
  b.pop(0);
  EXPECT_EQ(0, b.size());
}

TEST_F(TestBag, TestPop3){
  s::vector<uint> indexes;
  indexes.push_back(5);
  indexes.push_back(2);
  indexes.push_back(7);
  s::sort(s::begin(indexes), s::end(indexes), s::greater<uint>());

  for (size_t i = 0; i < 20; ++i){
    bs obj; obj.set(i);
    b.push_back(obj);
  }

  for (size_t i = 0; i < indexes.size(); ++i)
    b.pop(indexes[i]);

  ASSERT_EQ(17, b.size());

  for (size_t i = 0; i < b.size(); ++i){
    EXPECT_FALSE(b[i].test(5) == true);
    EXPECT_FALSE(b[i].test(2) == true);
    EXPECT_FALSE(b[i].test(7) == true);
  }
}

TEST_F(TestBag, TestPop4){
  s::vector<uint> indexes;
  indexes.push_back(17);
  indexes.push_back(18);
  indexes.push_back(19);
  s::sort(s::begin(indexes), s::end(indexes), s::greater<uint>());

  for (size_t i = 0; i < 20; ++i){
    bs obj; obj.set(i);
    b.push_back(obj);
  }

  for (size_t i = 0; i < indexes.size(); ++i)
    b.pop(indexes[i]);

  ASSERT_EQ(17, b.size());

  for (size_t i = 0; i < b.size(); ++i){
    EXPECT_FALSE(b[i].test(17) == true);
    EXPECT_FALSE(b[i].test(18) == true);
    EXPECT_FALSE(b[i].test(19) == true);
  }
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
