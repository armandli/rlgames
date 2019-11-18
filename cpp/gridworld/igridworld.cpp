#include <iostream>
#include <gridworld.h>

using namespace std;
using namespace gridworld;

int main(int argc, char* argv[]){
  uint dp[] = {4, 1, 1, 1, 0};

  for (int i = 1; i < argc; ++i){
    int p = atoi(argv[i]);
    dp[i - 1] = p;
  }

  Action action_map[] = {Action::UP, Action::DN, Action::LF, Action::RT};

  GridWorld world(dp[0], dp[1], dp[2], dp[3], dp[4]);
  world.print(cout);

  while (not world.is_complete()){
    uint aidx; cin >> aidx;
    assert(aidx < sizeof(action_map));
    world.move(action_map[aidx]);

    world.print(cout);
  }
  int reward = world.get_reward();
  s::cout << "Reward: " << reward << s::endl;
}
