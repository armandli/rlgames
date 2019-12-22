#include <gridworld.h>
#include <gridworld_models.h>

#include <iostream>

namespace s = std;
namespace g = gridworld;
namespace m = gridworld_pt;

int main(int argc, char* argv[]){
  m::GridEnv env(16, m::GridEnvMode::RandomMaze);
  g::GridWorld world = env.create();
  s::cout << world << s::endl;
}
