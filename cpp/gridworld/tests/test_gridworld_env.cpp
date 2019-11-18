#include <gtest/gtest.h>

#include <gridworld.h>
#include <gridworld_models.h>

namespace s = std;
namespace g = gridworld;
namespace t = torch;
namespace m = gridworld_pt;

//make sure it is a different environment every time
TEST(TestGridEnv, TestRandomSimple1){
  m::GridEnv env(4, m::GridEnvMode::RandomSimple);

  g::GridWorld g1 = env.create();
  const g::GridState& s1 = env.get_state(g1);
  g::GridWorld g2 = env.create();
  const g::GridState& s2 = env.get_state(g2);

  for (uint i = 0; i < 4 * 4; ++i){
    g::Obj o1 = s1.get(i);
    g::Obj o2 = s2.get(i);
    if (o1 != g::Obj::Empty || o2 != g::Obj::Empty)
      EXPECT_TRUE(o1 != o2);
  }
}
