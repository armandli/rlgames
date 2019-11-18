#include <gridworld.h>
#include <gridworld_models.h>
#include <gridworld_simulation.h>
#include <learning_metaparam.h>
#include <naive_qlearning.h>

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
  m::qlearning_metaparams<m::epsilon_greedy_metaparams, m::experience_replay_metaparams> mp;
  mp.epochs = 1000;
  mp.gamma = 0.9;
  mp.exp.epsilon = 1.;
  mp.max_steps = env.state_size() / 4 / 2;
  s::vector<float> losses;

  //maximum steps per game need to have close bound limit otherwise agent may spend too much time
  //getting discouraged for not able to find winning goal
  m::naive_qlearning<decltype(env), decltype(rlm), g::GridWorld, g::Action>(
    env,
    rlm,
    device,
    mp,
    losses
  );

  m::simulate_gridworld(env, rlm, env.state_size() / 4 / 2, device, true);

  if (losses.size() > 0){
    s::cout << "Final Loss: " << losses.back() << s::endl;
  }

  int sum = 0;
  int win_count = 0;
  int count = 100;
  for (int i = 0; i < count; ++i){
    int r = m::simulate_gridworld(env, rlm, env.state_size() / 4 / 2, device, false);
    sum += r;
    if (r > 1)
      win_count++;
  }
  s::cout << "Average Reward per 100 games: " << ((float)sum / (float)count) << " win percentage: " << ((float)win_count / (float)count) << s::endl;
}
