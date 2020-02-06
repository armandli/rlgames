#include <gridworld.h>
#include <gridworld_models.h>
#include <gridworld_simulation.h>
#include <learning_util.h>
#include <learning_metaparam.h>
#include <ppo_learning.h>

#include <torch/torch.h>

#include <ctime>
#include <vector>
#include <iostream>

namespace s = std;
namespace g = gridworld;
namespace t = torch;
namespace m = gridworld_pt;

//batch Proximal Policy Optimization with Curiosity Driven Exploration

int main(int argc, char* argv[]){
  uint grid_size = 16;
  if (argc > 1)
    grid_size = atoi(argv[1]);

  t::Device device(t::kCPU);
  if (t::cuda::is_available()){
    s::cout << "Using GPU" << s::endl;
    device = t::Device(t::kCUDA);
  }

  //m::GridEnv env(grid_size, m::GridEnvMode::RandomComplex, false /*step discount*/, true /*history discount*/);
  m::GridEnv env(grid_size, m::GridEnvMode::RandomSimple, false /*step discount*/, true /*history discount*/);
  //m::GridEnv env(grid_size, m::GridEnvMode::StaticSimple, false /*step discount*/);
  m::GridStateConvEncoder state_encoder(env);
  m::GridActionEncoder action_encoder(env);

  m::RLModel<m::SimplePPOICMModel, m::GridStateConvEncoder, m::GridActionEncoder, t::optim::Adam> rlm(
    m::SimplePPOICMModel(state_encoder.state_size(), m::Dim(4,5,5), m::Dim(6,3,3), m::Dim(4,5,5), m::Dim(6,3,3), 128, 160, 160, action_encoder.action_size()),
    s::move(state_encoder),
    s::move(action_encoder),
    1e-5F // learning rate
  );
  m::ppo_metaparams<m::simple_icm_metaparams> mp;
  mp.epochs = 12000;
  mp.batchsize = 16;
  mp.gamma = 0.99;
  mp.epsilon = 0.05; //PPO clip epsilon
  mp.tc_steps = 500;
  mp.max_steps = grid_size * grid_size / (grid_size / 2);
  mp.exp.beta = 0.2;
  mp.exp.eta = 100.F;
  mp.exp.epsilon = 1.F; //not used
  s::vector<float> losses;

  m::ppo_learning<decltype(env), decltype(rlm), g::GridWorld, g::Action>(
    env,
    rlm,
    device,
    mp,
    losses,
    time(NULL)
  );

  m::simulate_gridworld_ac(env, rlm, mp.max_steps, device, true);

  int sum = 0;
  int win_count = 0;
  int count = 100;
  for (int i = 0; i < count; ++i){
    int r = m::simulate_gridworld_ac(env, rlm, mp.max_steps, device, false);
    sum += r;
    if (r > 1)
      win_count++;
  }
  s::cout << "Average Reward per 100 games: " << ((float)sum / (float)count) << " win percentage: " << ((float)win_count / (float)count) << s::endl;
}
