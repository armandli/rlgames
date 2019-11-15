#include <gtest/gtest.h>

#include <gridworld.h>
#include <gridworld_models.h>
#include <experience.h>

#include <iostream>

namespace s = std;
namespace g = gridworld;
namespace t = torch;
namespace m = gridworld_pt;

class TestModelImpl : public t::nn::Module {
  t::nn::Linear l1, l2, l3;
public:
  TestModelImpl(sint64 isz, sint64 l1sz, sint64 l2sz, sint64 osz):
    l1(register_module("l1", t::nn::Linear(isz, l1sz))),
    l2(register_module("l2", t::nn::Linear(l1sz, l2sz))),
    l3(register_module("l3", t::nn::Linear(l2sz, osz)))
  {}

  t::Tensor forward(t::Tensor x){
    x = t::relu(l1(x));
    x = t::relu(l2(x));
    x = l3(x);
    return x;
  }
};
TORCH_MODULE(TestModel);

// test if experience system is functionally correct

TEST(TestExperience, TestExperienceUniqueness1){
  m::GridEnv env(4, m::GridEnvMode::RandomSimple);
  TestModel model(env.state_size(), 164, 150, env.action_size());
  m::GridStateEncoder sencoder(env);
  m::GridActionEncoder aencoder(env);

  uint buffer_size = 10;
  m::ExpReplayBuffer<m::Exp<g::Action>> replay_buffer(buffer_size);

  for (uint i = 0; i < buffer_size; ++i){
    g::GridWorld ins = env.create();
    t::Tensor tstate = sencoder.encode_state(ins.get_state());
    t::Tensor qval = model->forward(tstate);
    g::Action action = aencoder.decode_action(qval);
    env.apply_action(ins, action);
    t::Tensor tnstate = sencoder.encode_state(ins.get_state());
    float reward = ins.get_reward();
    replay_buffer.append(m::Exp<g::Action>(tstate, action, reward, tnstate));
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

  s::cout << tstatemap << s::endl;
  //s::cout << tnstatemap << s::endl;
}
