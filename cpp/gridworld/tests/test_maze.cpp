#include <gridworld.h>
#include <gridworld_models.h>

#include <iostream>

namespace s = std;
namespace g = gridworld;
namespace m = gridworld_pt;

int main(int argc, char* argv[]){
  int maze_size = 16;
  if (argc > 1){
    maze_size = atoi(argv[1]);
  }

  m::GridEnv env(maze_size, m::GridEnvMode::RandomMaze);
  g::GridWorld world = env.create();
  s::cout << world << s::endl;
}
