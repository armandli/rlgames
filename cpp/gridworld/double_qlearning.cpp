#include <gridworld.h>
#include <gridworld_models.h>
#include <learning_metaparam.h>
#include <double_qlearning.h>
#include <gridworld_simulation.h>

#include <torch/torch.h>

#include <ctime>
#include <vector>
#include <iostream>

namespace s = std;
namespace g = gridworld;
namespace t = torch;
namespace m = gridworld_pt;

//double q learning using random grid world of size k

int main(int argc, char* argv[]){
  uint grid_size = 4;
  if (argc > 1)
    grid_size = atoi(argv[1]);

  t::Device device(t::kCPU);
  if (t::cuda::is_available()){
    s::cout << "using GPU" << s::endl;
    device = t::Device(t::kCUDA);
  }

  m::GridEnv env(grid_size, m::GridEnvMode::RandomSimple);
  //m::GridEnv env(grid_size, m::GridEnvMode::StaticSimple);
  m::GridStateEncoder state_encoder(env);
  m::GridActionEncoder action_encoder(env);
  uint state_size = state_encoder.state_size().flatten_size();

  m::RLModel<m::SimpleQModel, m::GridStateEncoder, m::GridActionEncoder, t::optim::Adam> rlm(
    m::SimpleQModel(state_size, 164, 150, action_encoder.action_size()),
    s::move(state_encoder),
    s::move(action_encoder),
    1e-3F // learning rate
  );
  m::qlearning_metaparams<m::epsilon_greedy_metaparams, m::experience_replay_metaparams> mp;
  mp.epochs = 5000;
  mp.gamma = 0.9;
  mp.tc_steps = 500;
  mp.max_steps = state_size / 4 / 2;
  mp.exp.epsilon = 1.;
  mp.erb.sz = 2000;
  mp.erb.batchsize = 250;
  s::vector<float> losses;

  m::double_qlearning<decltype(env), decltype(rlm), g::GridWorld, g::Action>(
    env,
    rlm,
    device,
    mp,
    losses,
    time(NULL)
  );

  m::simulate_gridworld(env, rlm, state_size / 4 / 2, device, true);

  if (losses.size() > 0){
    s::cout << "Final Loss: " << losses.back() << s::endl;
  }

  int sum = 0;
  int win_count = 0;
  int count = 100;
  for (int i = 0; i < count; ++i){
    int r = m::simulate_gridworld(env, rlm, state_size / 4 / 2, device, false);
    sum += r;
    if (r > 1)
      win_count++;
  }
  s::cout << "Average Reward per 100 games: " << ((float)sum / (float)count) << " win percentage: " << ((float)win_count / (float)count) << s::endl;
}
