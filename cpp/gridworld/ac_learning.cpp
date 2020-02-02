#include <gridworld.h>
#include <gridworld_models.h>
#include <gridworld_simulation.h>
#include <learning_metaparam.h>
#include <ac_learning.h>

#include <torch/torch.h>

#include <ctime>
#include <vector>
#include <iostream>

namespace s = std;
namespace g = gridworld;
namespace t = torch;
namespace m = gridworld_pt;

//batch Actor-Critic Algorithm

int main(int argc, char* argv[]){
  uint grid_size = 4;
  if (argc > 1)
    grid_size = atoi(argv[1]);

  t::Device device(t::kCPU);
  if (t::cuda::is_available()){
    s::cout << "Using GPU" << s::endl;
    device = t::Device(t::kCUDA);
  }

  m::GridEnv env(grid_size, m::GridEnvMode::RandomSimple, false /*step discount*/);
  //m::GridEnv env(grid_size, m::GridEnvMode::StaticSimple, false /*step discount*/);
  m::GridStateEncoder state_encoder(env);
  m::GridActionEncoder action_encoder(env);
  uint state_size = state_encoder.state_size().flatten_size();

  m::RLModel<m::SimpleActorCriticModel, m::GridStateEncoder, m::GridActionEncoder, t::optim::Adam> rlm(
    m::SimpleActorCriticModel(state_size, 128, 64, 64, action_encoder.action_size(), 1),
    s::move(state_encoder),
    s::move(action_encoder),
    1e-5F // learning rate
  );
  m::ac_metaparams mp;
  mp.epochs = 10000;
  mp.batchsize = 16;
  mp.gamma = 0.9;
  mp.max_steps = state_size / 4;
  mp.tc_steps = 500;
  s::vector<float> losses;

  m::ac_learning<decltype(env), decltype(rlm), g::GridWorld, g::Action>(
    env,
    rlm,
    device,
    mp,
    losses,
    time(NULL)
  );

  m::simulate_gridworld_ac(env, rlm, state_size / 4, device, true);

  int sum = 0;
  int win_count = 0;
  int count = 100;
  for (int i = 0; i < count; ++i){
    int r = m::simulate_gridworld_ac(env, rlm, state_size / 4, device, false);
    sum += r;
    if (r > 1)
      win_count++;
  }
  s::cout << "Average Reward per 100 games: " << ((float)sum / (float)count) << " win percentage: " << ((float)win_count / (float)count) << s::endl;
}
