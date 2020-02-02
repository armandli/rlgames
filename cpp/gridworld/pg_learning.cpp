#include <gridworld.h>
#include <gridworld_models.h>
#include <gridworld_simulation.h>
#include <learning_metaparam.h>
#include <pg_learning.h>
#include <export_util.h>

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

  m::GridEnv env(grid_size, m::GridEnvMode::RandomSimple, false /*no step discount*/);
  //m::GridEnv env(grid_size, m::GridEnvMode::StaticSimple, false /*no step discount*/);
  m::GridStateEncoder state_encoder(env);
  m::GridActionEncoder action_encoder(env);
  uint state_size = state_encoder.state_size().flatten_size();

  m::RLModel<m::SimplePolicyModel, m::GridStateEncoder, m::GridActionEncoder, t::optim::Adam> rlm(
    m::SimplePolicyModel(state_size, 164, 150, action_encoder.action_size()),
    s::move(state_encoder),
    s::move(action_encoder),
    1e-5F // learning rate
  );
  m::pg_metaparams mp;
  mp.epochs = 8000;
  mp.batchsize = 8;
  mp.gamma = 0.99;
  mp.max_steps = state_size / 4;
  s::vector<float> losses;

  m::pg_learning<decltype(env), decltype(rlm), g::GridWorld, g::Action>(
    env,
    rlm,
    device,
    mp,
    losses,
    time(NULL)
  );

  m::simulate_gridworld(env, rlm, state_size / 4, device, true);

  int sum = 0;
  int win_count = 0;
  int count = 100;
  for (int i = 0; i < count; ++i){
    int r = m::simulate_gridworld(env, rlm, state_size / 4, device, false);
    sum += r;
    if (r > 1)
      win_count++;
  }
  s::cout << "Average Reward per 100 games: " << ((float)sum / (float)count) << " win percentage: " << ((float)win_count / (float)count) << s::endl;

  t::save(rlm.model, "pg_learning.pt");
  m::save_loss_array("pg_learning_train_loss.json", losses);
}
