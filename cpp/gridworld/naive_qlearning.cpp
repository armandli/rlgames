#include <gridworld.h>
#include <gridworld_models.h>
#include <naive_qlearning.h>
#include <gridworld_simulation.h>

#include <torch/torch.h>

#include <ctime>
#include <vector>
#include <iostream>

namespace s = std;
namespace g = gridworld;
namespace t = torch;
namespace m = gridworld_pt;

//naive q learning using static grid world of size k

int main(int argc, char* argv[]){
  uint grid_size = 4;
  if (argc > 1)
    grid_size = atoi(argv[1]);

  t::Device device(t::kCPU);
  if (t::cuda::is_available()){
    s::cout << "Use GPU" << s::endl;
    device = t::Device(t::kCUDA);
  }

  m::GridEnv env(grid_size, m::GridEnvMode::StaticSimple);
  m::RLModel<m::SimpleGridModel, m::GridStateEncoder, m::GridActionEncoder, t::optim::Adam> rlm(
    m::SimpleGridModel(env.state_size(), 164, 150, env.action_size()),
    m::GridStateEncoder(env),
    m::GridActionEncoder(env),
    1e-4F
  );
  s::vector<float> losses;

  //maximum steps per game need to have close bound limit otherwise agent may spend too much time
  //getting discouraged for not able to find winning goal
  m::naive_qlearning<decltype(env), decltype(rlm), g::GridWorld, g::Action>(
    env,
    rlm,
    1000,
    0.9,
    1.,
    losses,
    env.state_size() / 4,
    device
  );

  m::simulate_gridworld(env, rlm, env.state_size() / 4, device);

  s::cout << "Final Loss: " << losses.back() << s::endl;
}
