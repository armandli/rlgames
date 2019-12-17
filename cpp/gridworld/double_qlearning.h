#ifndef GRIDWORLD_DOUBLE_QLEARNING
#define GRIDWORLD_DOUBLE_QLEARNING

#include <learning_util.h>
#include <learning_metaparam.h>
#include <experience.h>

#include <torch/torch.h>

#include <cassert>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

//Double Q Learning with epsilon greedy exploration
//Optimization 1: Q function returns Q value for all actions of the same state
//instead of taking state and action as input and predict Q value
//Optimization 2: Use a separate target network for stabilization
//Optimization 3: Uses experience replay buffer
//Optimization 4: Use current network for action selection, use target network for its action value

template <typename ENV, typename RLM, typename INS, typename ACTION, typename ERB, uint loss_sampling_interval = 100>
void double_qlearning(
  ENV& env,
  RLM& rlm,
  t::Device device,
  qlearning_metaparams<epsilon_greedy_metaparams, experience_replay_metaparams>& mp,
  s::vector<float>& losses,
  uint64 random_seed){
  s::uniform_real_distribution<double> dist(0., 1.);
  s::uniform_int_distribution<uint> rand_action(0U, env.action_size() - 1);
  s::default_random_engine reng(random_seed);

  ExpReplayBuffer<Exp<ACTION>> replay_buffer(mp.erb.sz);

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
        ExpBatch<Exp<ACTION>> batch = replay_buffer.sample_batch(mp.erb.batchsize);
        t::Tensor output_dev = t::zeros({mp.erb.batchsize, env.action_size()}, device);
        t::Tensor target_dev = t::zeros({mp.erb.batchsize, env.action_size()}, device);
        uint h = 0;
        for (Exp<ACTION>& exp : batch){
          t::Tensor oqval_dev = rlm.model->forward(exp.tstate);
          t::Tensor nqval_dev = targetn->forward(exp.ntstate);
          t::Tensor nqselect_dev = rlm.model->forward(exp.ntstate);
          int argmaxq = t::argmax(nqselect_dev).item().to<int>();
          float maxq = nqval_dev[argmaxq].item().to<float>();
          t::Tensor y_dev = oqval_dev.clone();
          if (not exp.ntstate_isterminal){
            y_dev[(uint)exp.action] = t::scalar_tensor(exp.reward + mp.gamma * maxq, device);
          } else {
            y_dev[(uint)exp.action] = t::scalar_tensor(exp.reward, device);
          }
          output_dev[h] = oqval_dev;
          target_dev[h] = y_dev;
          h++;
        }
        t::Tensor loss_dev = t::mse_loss(output_dev, target_dev.detach());
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


#endif//GRIDWORLD_DOUBLE_QLEARNING
