#include <gridworld.h>
#include <gridworld_models.h>
#include <gridworld_simulation.h>
#include <learning_metaparam.h>
#include <pg_learning.h>

#include <torch/torch.h>

#include <ctime>
#include <vector>
#include <iostream>
#include <algorithm>

namespace s = std;
namespace g = gridworld;
namespace t = torch;
namespace m = gridworld_pt;

//REINFORCE algorithm; Monte Carlo Policy Gradient algorithm

int main(int argc, char* argv[]){
  uint grid_size = 4;
  if (argc > 1)
    grid_size = atoi(argv[1]);

  t::Device device(t::kCPU);
  if (t::cuda::is_available()){
    s::cout << "Using GPU" << s::endl;
    device = t::Device(t::kCUDA);
  }

  //m::GridEnv env(grid_size, m::GridEnvMode::RandomSimple, false /*no step discount*/);
  m::GridEnv env(grid_size, m::GridEnvMode::StaticSimple, false /*no step discount*/);

  m::RLModel<m::SimplePolicyModel, m::GridStateEncoder, m::GridActionEncoder, t::optim::Adam> rlm(
    m::SimplePolicyModel(env.state_size(), 164, 150, env.action_size()),
    m::GridStateEncoder(env),
    m::GridActionEncoder(env),
    1e-5F // learning rate
  );
  m::pg_metaparams mp;
  mp.epochs = 20000;
  mp.gamma = 0.99;
  mp.max_steps = env.state_size() / 4;
  s::vector<float> losses;

  m::pg_learning<decltype(env), decltype(rlm), g::GridWorld, g::Action>(
    env,
    rlm,
    device,
    mp,
    losses,
    time(NULL)
  );

  m::simulate_gridworld(env, rlm, env.state_size() / 4, device, true);

  uint non_zero_loss_count = std::accumulate(losses.begin(), losses.end(), 0U, [](uint acc, float v){ return acc + (s::abs(v) > 1e-7F); });
  s::cout << "Number of times goal/sink reached: " << non_zero_loss_count << s::endl;

  int sum = 0;
  int win_count = 0;
  int count = 100;
  for (int i = 0; i < count; ++i){
    int r = m::simulate_gridworld(env, rlm, env.state_size() / 4, device, false);
    sum += r;
    if (r > 1)
      win_count++;
  }
  s::cout << "Average Reward per 100 games: " << ((float)sum / (float)count) << " win percentage: " << ((float)win_count / (float)count) << s::endl;
}
