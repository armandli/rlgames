#include <gtest/gtest.h>

#include <torch/torch.h>

#include <gridworld.h>
#include <gridworld_models.h>
#include <experience.h>

#include <ctime>
#include <iostream>

namespace s = std;
namespace g = gridworld;
namespace t = torch;
namespace m = gridworld_pt;

// test if experience system is functionally correct

TEST(TestExperience, TestExperienceUniqueness1){
  t::Device device(t::kCPU);
  m::GridEnv env(4, m::GridEnvMode::RandomSimple);
  m::GridStateEncoder sencoder(env);
  m::GridActionEncoder aencoder(env);

  s::uniform_int_distribution<uint> rand_action(0U, env.action_size() - 1);
  s::default_random_engine reng(time(NULL));

  uint buffer_size = 10;
  m::ExpReplayBuffer<m::Exp<g::Action>> replay_buffer(buffer_size);

  for (uint i = 0; i < buffer_size; ++i){
    g::GridWorld ins = env.create();
    t::Tensor tstate = sencoder.encode_state(ins.get_state(), device);

    g::Action action = (g::Action)rand_action(reng);

    env.apply_action(ins, action);
    t::Tensor tnstate = sencoder.encode_state(ins.get_state(), device);

    float reward = ins.get_reward();
    bool is_complete = env.is_termination(ins);
    replay_buffer.append(m::Exp<g::Action>(tstate, action, reward, tnstate, is_complete));
  }
  m::ExpBatch<m::Exp<g::Action>> batch = replay_buffer.sample_batch(buffer_size);

  t::Tensor tstatemap = t::zeros({buffer_size, env.state_size()});
  t::Tensor tnstatemap = t::zeros({buffer_size, env.state_size()});
  uint h = 0;
  for (m::Exp<g::Action>& exp : batch){
    tstatemap[h] = exp.tstate;
    tnstatemap[h] = exp.ntstate;
    h++;
  }

  auto is_same = tstatemap == tnstatemap;

  s::cout << "Same:" << s::endl;
  s::cout << is_same << s::endl;

  s::cout << "State:" << s::endl;
  s::cout << tstatemap << s::endl;
  s::cout << "Next State:" << s::endl;
  s::cout << tnstatemap << s::endl;
}
