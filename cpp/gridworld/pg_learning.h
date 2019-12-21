#ifndef GRIDWORLD_PG_LEARNING
#define GRIDWORLD_PG_LEARNING

#include <learning_util.h>
#include <learning_metaparam.h>

#include <cmath>
#include <vector>
#include <random>

#include <torch/torch.h>

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

template <typename ENV, typename RLM, typename INS, typename ACTION, uint loss_sampling_interval = 100>
void pg_learning(
  ENV& env,
  RLM& rlm,
  t::Device device,
  pg_metaparams& mp,
  s::vector<float>& losses,
  uint64 random_seed) {

  t::Device cpu_device(t::kCPU);
  s::default_random_engine reng(random_seed);

  rlm.model->to(device);

  for (uint64 i = 0; i < mp.epochs; ++i){
    INS ins = env.create();
    uint64 step_count = 0;
    t::Tensor action_dev = t::zeros({(uint)mp.max_steps}, device);
    s::vector<float> reward_vec(mp.max_steps);
    float reward_accum = 0.0F;
    while (not env.is_termination(ins) && step_count < mp.max_steps){
      t::Tensor tstate_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
      t::Tensor tadist_dev = rlm.model->forward(tstate_dev);
      t::Tensor tadist = tadist_dev.to(cpu_device);
      ACTION action = (ACTION)sample_discrete_distribution(tadist.data_ptr<float>(), env.action_size(), reng);
      env.apply_action(ins, action);
      action_dev[step_count] = tadist_dev[(uint)action];
      reward_accum += env.get_reward(ins);
      reward_vec[step_count] = reward_accum;
      step_count++;
    }
    s::vector<float> discount_vec(step_count);
    //higher discount on earlier action than later action
    for (uint64 i = step_count - 1, pow = 0; i < step_count; --i, ++pow)
      discount_vec[i] = s::pow(mp.gamma, pow);
    t::Tensor discount = t::from_blob(discount_vec.data(), {(uint)step_count});
    t::Tensor discount_dev = discount.to(device);
    //rewards are supposed to be sum of future rewards, not historically accumulated rewards
    for (uint64 i = step_count - 2; i < mp.max_steps; --i)
      reward_vec[i] = reward_vec[i+1] - reward_vec[i];
    t::Tensor reward = t::from_blob(reward_vec.data(), {(uint)step_count});
    t::Tensor reward_dev = reward.to(device);
    reward_dev = discount_dev * reward_dev;
    //to reduce variance, we normalize the reward value
    reward_dev = (reward_dev - reward_dev.mean()) / (reward_dev.std() + 1e-07);
    action_dev = action_dev.slice(0, 0, step_count);
    t::Tensor loss_dev = t::mean(-1.F * reward_dev.detach() * t::log(action_dev));

    loss_dev.backward();
    rlm.optimizer.step();

    if (i % loss_sampling_interval == 0){
      s::cout << "loss: " << loss_dev.item().to<float>() << s::endl;
      losses.push_back(loss_dev.item().to<float>());
    }
  }
}

} //gridworld_pt

#endif//GRIDWORLD_PG_LEARNING
