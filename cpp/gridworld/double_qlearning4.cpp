#include <gridworld.h>
#include <gridworld_models.h>
#include <learning_util.h>
#include <learning_metaparam.h>
#include <double_qlearning4.h>
#include <gridworld_simulation.h>

#include <torch/torch.h>

#include <ctime>
#include <vector>
#include <iostream>

namespace s = std;
namespace g = gridworld;
namespace t = torch;
namespace m = gridworld_pt;

//double q learning with ICM using random grid world of size k

int main(int argc, char* argv[]){
  uint grid_size = 16;
  if (argc > 1)
    grid_size = atoi(argv[1]);

  t::Device device(t::kCPU);
  if (t::cuda::is_available()){
    s::cout << "using GPU" << s::endl;
    device = t::Device(t::kCUDA);
  }

  //m::GridEnv env(grid_size, m::GridEnvMode::RandomMaze, false /*step discount*/, true /*discourage historical move*/);
  m::GridEnv env(grid_size, m::GridEnvMode::RandomComplex, false /*step discount*/, true /*discourage historical move*/);
  //m::GridEnv env(grid_size, m::GridEnvMode::RandomSimple, false /*step discount*/);
  //m::GridEnv env(grid_size, m::GridEnvMode::StaticSimple);
  m::GridStateConvEncoder state_encoder(env);
  m::GridActionEncoder action_encoder(env);
  m::RLModel<m::SimpleICMQModel, m::GridStateConvEncoder, m::GridActionEncoder, t::optim::Adam> rlm(
    m::SimpleICMQModel(state_encoder.state_size(), m::Dim(4, 5, 5), m::Dim(6, 3, 3), m::Dim(4, 5, 5), m::Dim(6, 3, 3), grid_size * grid_size * 2, 128, action_encoder.action_size()),
    s::move(state_encoder),
    s::move(action_encoder),
    1e-5F //learning rate
  );
  m::qlearning_metaparams<m::simple_icm_metaparams, m::experience_replay_metaparams> mp;
  mp.epochs = 6000;
  mp.gamma = 0.99;
  mp.tc_steps = 500;
  mp.max_steps = grid_size * grid_size / (grid_size / 2);
  mp.exp.eta = 100.0; //really encourage exploration
  mp.exp.beta = 0.2;
  mp.exp.epsilon = 1.;
  mp.erb.sz = 4000;
  mp.erb.batchsize = 256;
  s::vector<float> losses;

  m::double_qlearning4<decltype(env), decltype(rlm), g::GridWorld, g::Action>(
    env,
    rlm,
    device,
    mp,
    losses,
    time(NULL)
  );

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
