#ifndef GRIDWORLD_DOUBLE_QLEARNING4
#define GRIDWORLD_DOUBLE_QLEARNING4

#include <learning_util.h>
#include <learning_debug.h>
#include <learning_metaparam.h>
#include <experience_util.h>
#include <experience3.h>

#include <torch/torch.h>

#include <cassert>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

//Double Q Learning with Curiosity Driven Exploration
//Optimization 1: Q function returns Q value for all actions of the same
//state
//Optimization 2: Use a separate target network for stabilization
//Optimization 3: Use experience replay buffer, improved
//Optimization 4: Use current network for action selection, target network
//for action value
//Optimization 5: use curiosity driven exploration

template <typename ENV, typename RLM, typename INS, typename ACTION, uint loss_sampling_interval = 100>
void double_qlearning4(
  ENV& env,
  RLM& rlm,
  t::Device device,
  qlearning_metaparams<simple_icm_metaparams, experience_replay_metaparams>& mp,
  s::vector<float>& losses,
  uint64 random_seed){

  s::uniform_real_distribution<double> dist(0., 1.);
  s::uniform_int_distribution<uint> rand_action(0U, rlm.action_encoder.action_size() - 1);
  s::default_random_engine reng(random_seed);

  ExpReplayBuffer3<ACTION> replay_buffer(mp.erb.sz, rlm.state_encoder.state_size(), device);

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

      t::Tensor state_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
      t::Tensor qval_dev = rlm.model->forward(state_dev);
      //we still need epsilon greedy in Q-learning even with ICM module
      ACTION action;
      if (dist(reng) < mp.exp.epsilon){
        action = (ACTION)rand_action(reng);
      } else {
        action = rlm.action_encoder.decode_action(qval_dev);
      }
      env.apply_action(ins, action);
      t::Tensor nstate_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
      float reward = env.get_reward(ins);
      replay_buffer.append(Exp<ACTION>(state_dev, action, reward, nstate_dev, env.is_termination(ins)));
      if (replay_buffer.is_filled()){
        rlm.model->zero_grad();

        ExpReplayBuffer3<ACTION> batch = replay_buffer.sample_batch(mp.erb.batchsize);
        ARTArray actions_rewards = batch.actions_n_rewards();

        // Q value computation
        t::Tensor action = t::from_blob(actions_rewards.actions.data(), {mp.erb.batchsize}, t::kLong);
        t::Tensor action_dev = action.to(device);
        t::Tensor oqval_dev = rlm.model->forward(batch.states_tensor());
        oqval_dev = t::index_select(oqval_dev, 1, action_dev).diagonal();

        // target Q value computation
        t::Tensor nqval_dev = targetn->forward(batch.nstates_tensor());
        t::Tensor nqselect_dev = rlm.model->forward(batch.nstates_tensor());
        nqselect_dev = t::argmax(nqselect_dev, 1);
        nqval_dev = t::index_select(nqval_dev, 1, nqselect_dev).diagonal();
        t::Tensor reward = t::from_blob(actions_rewards.rewards.data(), {mp.erb.batchsize});
        t::Tensor reward_dev = reward.to(device);
        t::Tensor terminal = t::from_blob(actions_rewards.is_terminals.data(), {mp.erb.batchsize}, t::kLong);
        t::Tensor terminal_dev = terminal.to(device).logical_not();
        t::Tensor target_dev = reward_dev + mp.gamma * terminal_dev * nqval_dev;

        //ICM loss
        //one-hot-encode action vector
        t::Tensor ones_dev = t::ones({mp.erb.batchsize, 1}, device);
        action_dev = action_dev.reshape({mp.erb.batchsize, 1});
        t::Tensor action_onehot_dev = t::zeros({mp.erb.batchsize, rlm.action_encoder.action_size()}, device).scatter_(1, action_dev, ones_dev);
        t::Tensor target_nstate_dev = rlm.model->featurize_state(batch.nstates_tensor());
        t::Tensor fnstate_dev = rlm.model->icm_forward_dynamics(batch.states_tensor(), action_onehot_dev);
        t::Tensor iaction_dev = rlm.model->icm_inverse_dynamics(batch.states_tensor(), batch.nstates_tensor());

        //total loss
        t::Tensor intrinsic_reward = mp.exp.beta * t::mse_loss(fnstate_dev, target_nstate_dev.detach());
        t::Tensor id_loss = t::mean(t::sum(-1.F * action_onehot_dev.detach() * t::log(iaction_dev), -1));
        t::Tensor extrinsic_loss = t::mse_loss(oqval_dev, target_dev.detach());
        t::Tensor loss_dev = extrinsic_loss + intrinsic_reward + id_loss;

//        s::cout << "extrinsic loss: " << extrinsic_loss.item().to<float>() << s::endl;
//        s::cout << "Intrinsic reward: "<< t::mean(intrinsic_reward).item().to<float>() << s::endl;
//        s::cout << "Inverse Dynaics loss: " << id_loss.item().to<float>() << s::endl;

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

#endif//GRIDWORLD_DOUBLE_QLEARNING4
