#ifndef GRIDWORLD_DOUBLE_QLEARNING
#define GRIDWORLD_DOUBLE_QLEARNING

#include <pytorch_util.h>
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
  qlearning_metaparams<epsilon_greedy_metaparams, experience_replay_metaparams> mp,
  s::vector<float>& losses){

  t::Device cpu_device(t::kCPU);

  s::uniform_real_distribution<double> dist(0., 1.);
  s::uniform_int_distribution<uint> rand_action(0U, env.action_size() - 1);
  s::default_random_engine reng;

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

      t::Tensor tstate = rlm.state_encoder.encode_state(env.get_state(ins));
      t::Tensor tstate_dev = tstate.to(device);
      t::Tensor qval_dev = rlm.model->forward(tstate_dev);
      ACTION action;
      if (dist(reng) < mp.exp.epsilon){
        action = (ACTION)rand_action(reng);
      } else {
        t::Tensor qval = qval_dev.to(cpu_device);
        action = rlm.action_encoder.decode_action(qval);
      }
      env.apply_action(ins, action);
      t::Tensor tnstate = rlm.state_encoder.encode_state(env.get_state(ins));
      float reward = env.get_reward(ins);
      replay_buffer.append(Exp<ACTION>(tstate, action, reward, tnstate, env.is_termination(ins)));
      if (replay_buffer.is_filled()){
        rlm.model->zero_grad();
        ExpBatch<Exp<ACTION>> batch = replay_buffer.sample_batch(mp.erb.batchsize);
        t::Tensor output_dev = t::zeros({mp.erb.batchsize, env.action_size()}, device);
        t::Tensor target = t::zeros({mp.erb.batchsize, env.action_size()});
        uint h = 0;
        for (Exp<ACTION>& exp : batch){
          t::Tensor tstate_dev = exp.tstate.to(device);
          t::Tensor oqval_dev = rlm.model->forward(tstate_dev);
          t::Tensor oqval = oqval_dev.to(cpu_device);

          t::Tensor ntstate_dev = exp.ntstate.to(device);
          t::Tensor nqval_dev = targetn->forward(ntstate_dev);
          t::Tensor nqval = nqval_dev.to(cpu_device);

          t::Tensor nqselect_dev = rlm.model->forward(ntstate_dev);
          t::Tensor nqselect = nqselect_dev.to(cpu_device);

          int argmaxq = nqselect.argmax().item().to<int>();
          float maxq = nqval[argmaxq].item().to<float>();

          t::Tensor y = oqval.clone();
          if (not exp.ntstate_isterminal){
            y[(uint)exp.action] = t::scalar_tensor(exp.reward + mp.gamma * maxq);
          } else {
            y[(uint)exp.action] = t::scalar_tensor(exp.reward);
          }
          output_dev[h] = oqval_dev;
          target[h] = y;
          h++;
        }
        t::Tensor target_dev = target.to(device);
        t::Tensor loss_dev = t::mse_loss(output_dev, target_dev.detach());
        loss_dev.backward();
        rlm.optimizer.step();

        if ((i + step_count) % loss_sampling_interval == 0){
          t::Tensor loss = loss_dev.to(cpu_device);
          losses.push_back(loss.item().to<float>());
        }
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
