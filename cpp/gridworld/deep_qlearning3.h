#ifndef GRIDWORLD_DEEP_QLEARNING3
#define GRIDWORLD_DEEP_QLEARNING3

#include <learning_util.h>
#include <learning_metaparam.h>
#include <experience_util.h>
#include <prioritized_experience.h>

#include <torch/torch.h>

#include <cassert>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

//Classic Deep Q Learning with epsilon greedy exploration
//Optimization 1: Q function returns Q value for all actions of the same state
//instead of taking state and action as input and predict Q value
//Optimization 2: Use a separate target network for stabilization
//Optimization 3: Uses prioritized experience replay buffer

template <typename ENV, typename RLM, typename INS, typename ACTION, uint loss_sampling_interval = 100>
void deep_qlearning3(
  ENV& env,
  RLM& rlm,
  t::Device device,
  qlearning_metaparams<epsilon_greedy_metaparams, prioritized_experience_replay_metaparams>& mp,
  s::vector<float>& losses,
  uint64 random_seed){
  s::uniform_real_distribution<double> dist(0., 1.);
  s::uniform_int_distribution<uint> rand_action(0U, rlm.action_encoder.action_size() - 1);
  s::default_random_engine reng(random_seed);

  PriExpReplayBuffer<ACTION> replay_buffer(
    mp.erb.sz,
    rlm.state_encoder.state_size().flatten_size(),
    device,
    random_seed,
    mp.erb.alpha,
    mp.erb.init_delta);

  decltype(rlm.model) targetn(rlm.model);
  copy_state(targetn, rlm.model);

  rlm.model->to(device);
  targetn->to(device);

  for (uint64 i = 0, tc_step = 0; i < mp.epochs; ++i){
    INS ins = env.create();
    uint64 step_count = 0;
    while (not env.is_termination(ins) && step_count < mp.max_steps){
      //update targetn parameters
      if (tc_step >= mp.tc_steps){
        copy_state(targetn, rlm.model);
        tc_step = 0;
      }

      t::Tensor tstate_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
      t::Tensor qval_dev = rlm.model->forward(tstate_dev);
      ACTION action;
      if (dist(reng) < mp.exp.epsilon){
        action = (ACTION)rand_action(reng);
      } else {
        action = rlm.action_encoder.decode_action(qval_dev);
      }
      env.apply_action(ins, action);
      t::Tensor tnstate_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
      float reward = env.get_reward(ins);
      replay_buffer.append(Exp<ACTION>(tstate_dev, action, reward, tnstate_dev, env.is_termination(ins)));
      if (replay_buffer.is_filled()){
        rlm.model->zero_grad();

        PriExpReplayBatch<ACTION> batch = replay_buffer.sample_batch(mp.erb.batchsize);
        ARTArray actions_rewards = batch.actions_n_rewards();

        t::Tensor action = t::from_blob(actions_rewards.actions.data(), {mp.erb.batchsize}, t::kLong);
        t::Tensor action_dev = action.to(device);
        t::Tensor oqval_dev = rlm.model->forward(batch.states_tensor());
        oqval_dev = t::index_select(oqval_dev, 1, action_dev);
        oqval_dev = oqval_dev.diagonal();

        t::Tensor nqval_dev = targetn->forward(batch.nstates_tensor());
        nqval_dev = t::max_values(nqval_dev, 1);
        t::Tensor reward = t::from_blob(actions_rewards.rewards.data(), {mp.erb.batchsize});
        t::Tensor reward_dev = reward.to(device);
        t::Tensor terminal = t::from_blob(actions_rewards.is_terminals.data(), {mp.erb.batchsize}, t::kLong);
        t::Tensor terminal_dev = terminal.to(device);
        terminal_dev = terminal_dev.logical_not();
        t::Tensor target_dev = reward_dev + mp.gamma * terminal_dev * nqval_dev;
        t::Tensor loss_dev = t::mse_loss(oqval_dev, target_dev.detach());

        t::Tensor delta_dev = t::pow(target_dev - oqval_dev, 2);
        replay_buffer.update(batch, delta_dev);

        loss_dev.backward();
        rlm.optimizer.step();

        if ((i + step_count) % loss_sampling_interval == 0)
          losses.push_back(loss_dev.item().to<float>());
      }

      step_count++;
      tc_step++;
    }

    if (replay_buffer.is_filled()){
      //epsilon decay
      if (mp.exp.epsilon > 0.1)
        mp.exp.epsilon -= 1. / (double)mp.epochs;

      if (i % 10 == 0)
        s::cout << "epsilon: " << mp.exp.epsilon << s::endl;
    }
  }
}


} // gridworld_pt

#endif//GRIDWORLD_DEEP_QLEARNING3
