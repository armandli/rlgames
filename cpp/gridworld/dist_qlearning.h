#ifndef GRIDWORLD_DISTRIBUTIONAL_QLEARNING
#define GRIDWORLD_DISTRIBUTIONAL_QLEARNING

#include <learning_util.h>
#include <learning_metaparam.h>
#include <experience_util.h>
#include <experience2.h>

#include <torch/torch.h>

#include <cassert>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

//Distributional Q Learning with epsilon greedy exploration
//Optimization 1: Use target network for stabilization
//Optimization 2: Use experience replay buffer
//Optimization 3: do double q-learning

t::Tensor construct_nonterminal_support(
  s::vector<float>& buffer,
  const s::vector<float>& reward,
  const s::vector<float>& support,
  uint batch_size,
  uint reward_dist_slices,
  float vmax,
  float vmin,
  float zdelta,
  float gamma){
  for (float& v : buffer)
    v = 0.0F;
  for (uint i = 0; i < batch_size; ++i)
    for (uint j = 0; j < reward_dist_slices; ++j){
      float nv = (s::max(vmin, s::min(vmax, reward[i] + gamma * support[j])) - vmin) / zdelta;
      float ml = s::floor(nv);
      float mu = s::ceil(nv);
      buffer[i * reward_dist_slices + (uint)ml] += (nv - ml);
      buffer[i * reward_dist_slices + (uint)mu] += (mu - nv);
    }

  t::Tensor v = t::from_blob(buffer.data(), {batch_size, reward_dist_slices});
  return v.clone();
}

t::Tensor construct_terminal_reward(
  s::vector<float>& buffer,
  const s::vector<float>& reward,
  uint batch_size,
  uint reward_dist_slices,
  float vmax,
  float vmin,
  float zdelta){
  for (float& v : buffer)
    v = 0.0F;
  for (uint i = 0; i < batch_size; ++i){
    float nv = (s::max(vmin, s::min(vmax, reward[i])) - vmin) / zdelta;
    float ml = s::floor(nv);
    float mu = s::ceil(nv);
    buffer[i * reward_dist_slices + (uint)ml] = (nv - ml);
    buffer[i * reward_dist_slices + (uint)mu] = (mu - nv);
  }

  t::Tensor v = t::from_blob(buffer.data(), {batch_size, reward_dist_slices});
  return v.clone();
}

template <typename ENV, typename RLM, typename INS, typename ACTION, uint loss_sampling_interval = 100>
void distributional_qlearning(
  ENV& env,
  RLM& rlm,
  t::Device device,
  distqlearning_metaparams<epsilon_greedy_metaparams, experience_replay_metaparams>& mp,
  s::vector<float>& losses,
  uint64 random_seed){

  t::Device cpu_device(t::kCPU);

  s::uniform_real_distribution<double> dist(0., 1.);
  s::uniform_int_distribution<uint> rand_action(0U, env.action_size() - 1);
  s::default_random_engine reng(random_seed);

  //distribution construction buffer
  s::vector<float> dist_buffer(mp.erb.batchsize * mp.reward_dist_slices);

  //construct the value distribution support vector
  s::vector<float> support_vec(mp.reward_dist_slices);
  float zdelta = 1.F;
  {
    INS ins = env.create();
    zdelta = (env.max_reward(ins) - env.min_reward(ins)) / (float)mp.reward_dist_slices;
    for (uint i = 0; i < support_vec.size(); ++i)
      support_vec[i] = env.min_reward(ins) + i * zdelta;
  }
  t::Tensor support = t::from_blob(support_vec.data(), {mp.reward_dist_slices});
  t::Tensor support_dev = support.to(device);

  ExpReplayBuffer2<ACTION> replay_buffer(mp.erb.sz, env.state_size(), device);

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
      t::Tensor dqval_dev = rlm.model->forward(tstate_dev);
      ACTION action;
      if (dist(reng) < mp.exp.epsilon){
        action = (ACTION)rand_action(reng);
      } else {
        t::Tensor action_dev = (dqval_dev * support_dev).sum(-1);
        action = rlm.action_encoder.decode_action(action_dev);
      }
      env.apply_action(ins, action);
      t::Tensor tnstate_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
      float reward = env.get_reward(ins);
      replay_buffer.append(Exp<ACTION>(tstate_dev, action, reward, tnstate_dev, env.is_termination(ins)));
      if (replay_buffer.is_filled()){
        rlm.model->zero_grad();

        ExpReplayBuffer2<ACTION> batch = replay_buffer.sample_batch(mp.erb.batchsize);
        ARTArray actions_rewards = batch.actions_n_rewards();

        //output
        t::Tensor action = t::from_blob(actions_rewards.actions.data(), {mp.erb.batchsize}, t::kLong);
        t::Tensor action_dev = action.to(device);
        action_dev = action_dev.repeat_interleave(mp.reward_dist_slices).reshape({mp.erb.batchsize, 1, mp.reward_dist_slices});
        t::Tensor doqval_dev = rlm.model->forward(batch.states_tensor());
        doqval_dev = t::squeeze(t::gather(doqval_dev, 1, action_dev));

        //construct target distribution assuming every sample is non-terminal
        t::Tensor dnqval_dev = targetn->forward(batch.nstates_tensor());
        t::Tensor dnqselect_dev = rlm.model->forward(batch.nstates_tensor());
        dnqselect_dev = t::argmax((dnqselect_dev * support_dev).sum(-1), 1);
        dnqselect_dev = dnqselect_dev.repeat_interleave(mp.reward_dist_slices).reshape({mp.erb.batchsize, 1, mp.reward_dist_slices});
        dnqval_dev = t::squeeze(t::gather(dnqval_dev, 1, dnqselect_dev));
        t::Tensor ntsupport = construct_nonterminal_support(dist_buffer, actions_rewards.rewards, support_vec, mp.erb.batchsize, mp.reward_dist_slices, env.max_reward(ins), env.min_reward(ins), zdelta, mp.gamma);
        t::Tensor ntsupport_dev = ntsupport.to(device);
        t::Tensor nttarget_dev = dnqval_dev * ntsupport_dev;
        nttarget_dev = t::softmax(nttarget_dev, -1);

        //construct target distribution assuming every sample is terminal
        t::Tensor ttarget = construct_terminal_reward(dist_buffer, actions_rewards.rewards, mp.erb.batchsize, mp.reward_dist_slices, env.max_reward(ins), env.min_reward(ins), zdelta);
        t::Tensor ttarget_dev = ttarget.to(device);

        //merge the 2 tensor, gather and index on terminal flag
        t::Tensor terminal = t::from_blob(actions_rewards.is_terminals.data(), {mp.erb.batchsize}, t::kLong);
        t::Tensor terminal_dev = terminal.to(device);
        terminal_dev = terminal_dev.repeat_interleave(mp.reward_dist_slices);
        terminal_dev = terminal_dev.reshape({1, mp.erb.batchsize, mp.reward_dist_slices});
        nttarget_dev = nttarget_dev.reshape({1, mp.erb.batchsize, mp.reward_dist_slices});
        ttarget_dev = ttarget_dev.reshape({1, mp.erb.batchsize, mp.reward_dist_slices});
        t::Tensor target_dev = t::cat({nttarget_dev, ttarget_dev}, 0);
        target_dev = t::squeeze(t::gather(target_dev, 0, terminal_dev));

        //cross entropy loss
        t::Tensor loss_dev = t::mean(-1.F * target_dev.detach() * t::log(doqval_dev));

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

} //gridworld_pt


#endif//GRIDWORLD_DISTRIBUTIONAL_QLEARNING
