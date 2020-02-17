#include <gridworld.h>
#include <gridworld_models.h>
#include <learning_util.h>
#include <learning_metaparam.h>
#include <double_qlearning5.h>
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
  uint grid_size = 16;
  if (argc > 1)
    grid_size = atoi(argv[1]);

  t::Device device(t::kCPU);
  if (t::cuda::is_available()){
    s::cout << "using GPU" << s::endl;
    device = t::Device(t::kCUDA);
  }

  m::GridEnv env(grid_size, m::GridEnvMode::RandomRepeatedComplex, 1000, false /*step discount*/, false /*historical discount*/, false /*historical move termination*/);
  //m::GridEnv env(grid_size, m::GridEnvMode::RandomSimple, false /*not step discount*/, true /*historical discount*/, true /*historical termination*/);
  //m::GridEnv env(grid_size, m::GridEnvMode::RandomSimple, false /*not step discount*/, true /*historical discount*/);
  //m::GridEnv env(grid_size, m::GridEnvMode::StaticSimple);
  m::GridStateConvEncoder state_encoder(env);
  m::GridActionEncoder action_encoder(env);
  m::RLModel<m::SimpleConvQModel, m::GridStateConvEncoder, m::GridActionEncoder, t::optim::Adam> rlm(
    m::SimpleConvQModel(state_encoder.state_size(), m::Dim(4, 5, 5), m::Dim(6, 3 ,3), 164, action_encoder.action_size()),
    s::move(state_encoder),
    s::move(action_encoder),
    1e-5F // learning rate
  );
  m::qlearning_metaparams<m::epsilon_greedy_metaparams, m::experience_replay_metaparams> mp;
  mp.epochs = 8000;
  mp.gamma = 0.99;
  mp.tc_steps = 500;
  mp.max_steps = grid_size * grid_size;
  mp.exp.epsilon = 1.;
  mp.erb.sz = 8000;
  mp.erb.batchsize = 256;
  s::vector<float> losses;

  m::double_qlearning5<decltype(env), decltype(rlm), g::GridWorld, g::Action>(
    env,
    rlm,
    device,
    mp,
    losses,
    time(NULL)
  );

  env.set_repeat_count(1);
  m::simulate_gridworld(env, rlm, mp.max_steps, device, true);

  if (losses.size() > 0){
    s::cout << "Final Loss: " << losses.back() << s::endl;
  }

  int sum = 0;
  int win_count = 0;
  int count = 100;
  for (int i = 0; i < count; ++i){
    int r = m::simulate_gridworld(env, rlm, mp.max_steps, device, false);
    sum += r;
    if (r > 1)
      win_count++;
  }
  s::cout << "Average Reward per 100 games: " << ((float)sum / (float)count) << " win percentage: " << ((float)win_count / (float)count) << s::endl;
}
