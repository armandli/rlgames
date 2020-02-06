#ifndef GRIDWORLD_PPO_LEARNING
#define GRIDWORLD_PPO_LEARNING

#include <learning_util.h>
#include <learning_debug.h>
#include <learning_metaparam.h>
#include <experience_buffer2.h>

#include <cassert>
#include <vector>
#include <random>
#include <iostream>

#include <torch/torch.h>

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

//Proximal Policy Optimization with Curiosity Driven Exploration

template <typename ENV, typename RLM, typename INS, typename ACTION, uint loss_sampling_interval = 100>
void ppo_learning(
  ENV& env,
  RLM& rlm,
  t::Device device,
  ppo_metaparams<simple_icm_metaparams>& mp,
  s::vector<float>& losses,
  uint64 random_seed){

  t::Device cpu_device(t::kCPU);
  s::default_random_engine reng(random_seed);
  ExpBuffer2<ACTION> buffer(mp.batchsize * mp.max_steps, rlm.state_encoder.state_size(), device);

  bool is_first_loop = true;
  decltype(rlm.model) opi(rlm.model);
  decltype(rlm.model) targetn(rlm.model);
  copy_state(opi, rlm.model);
  copy_state(targetn, rlm.model);

  rlm.model->to(device);
  opi->to(device);

  for (uint64 i = 0, tc_step = 0; i < mp.epochs; ++i){
    //update targetn parameters
    if (tc_step >= mp.tc_steps){
      copy_state(targetn, rlm.model);
      tc_step = 0;
    }
    buffer.reset();

    //collecting sample batch using the current model
    for (uint64 b = 0; b < mp.batchsize; ++b, ++tc_step){
      INS ins = env.create();
      uint step_count = 0;
      while (not env.is_termination(ins) && step_count < mp.max_steps){
        t::Tensor state_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
        t::Tensor adist_dev = rlm.model->actor_forward(state_dev);
        t::Tensor adist = adist_dev.to(cpu_device);
        ACTION action = (ACTION)sample_discrete_distribution(adist.data_ptr<float>(), rlm.action_encoder.action_size(), reng);
        env.apply_action(ins, action);
        float reward = env.get_reward(ins);
        t::Tensor nstate_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
        buffer.append(Exp<ACTION>(state_dev, action, reward, nstate_dev, env.is_termination(ins)));
        step_count++;
      }
    }


    rlm.model->zero_grad();

    ARTArray actions_rewards = buffer.actions_n_rewards();
    t::Tensor action_dev = t::from_blob(actions_rewards.actions.data(), {(sint64)actions_rewards.actions.size(), 1}, t::kLong).to(device);
    t::Tensor ereward_dev = t::from_blob(actions_rewards.rewards.data(), {(sint64)actions_rewards.rewards.size()}).to(device);
    t::Tensor terminal_dev = t::from_blob(actions_rewards.is_terminals.data(), {(sint64)actions_rewards.is_terminals.size()}, t::kLong).to(device).logical_not();
    t::Tensor ones_dev = t::ones({(sint64)actions_rewards.actions.size(), 1}, device);

    // use the old policy and replace it with the new copy, order is important here
    t::Tensor oad_dev = opi->actor_forward(buffer.states_tensor());
    oad_dev = t::index_select(oad_dev, 1, action_dev.squeeze()).diagonal();
    if (not is_first_loop)
      copy_state(opi, rlm.model);
    else
      is_first_loop = false;

    ACTensor avs_dev = rlm.model->forward(buffer.states_tensor());
    avs_dev.actor_out = t::index_select(avs_dev.actor_out, 1, action_dev.squeeze()).diagonal();
    t::Tensor nvs_dev = targetn->critic_forward(buffer.nstates_tensor());

    //ICM forward dynamics
    t::Tensor action_onehot_dev = t::zeros({(sint64)actions_rewards.actions.size(), rlm.action_encoder.action_size()}, device).scatter_(1, action_dev, ones_dev);
    t::Tensor fnstate_dev = rlm.model->icm_forward_dynamics(buffer.states_tensor(), action_onehot_dev);

    //ICM inverse dynamics
    t::Tensor iaction_dev = rlm.model->icm_inverse_dynamics(buffer.states_tensor(), buffer.nstates_tensor());

    //ICM intrinsic reward
    t::Tensor target_nstate_dev = rlm.model->featurize_state(buffer.nstates_tensor());
    t::Tensor ireward_dev = mp.exp.eta * t::sum(t::pow(target_nstate_dev - fnstate_dev, 2), -1);

    //ICM F loss
    t::Tensor icmf_loss_dev = mp.exp.beta * t::mse_loss(fnstate_dev, target_nstate_dev.detach());

    //ICM I loss
    t::Tensor icmi_loss_dev = (1.F - mp.exp.beta) * t::mean(t::sum(-1.F * action_onehot_dev.detach() * t::log(iaction_dev), -1));

    //Advantage
    t::Tensor reward_dev = ereward_dev + ireward_dev;
    t::Tensor target_value_dev = reward_dev + mp.gamma * terminal_dev * nvs_dev;
    t::Tensor advantage_dev = target_value_dev - avs_dev.critic_out;

    //PPO Loss
    t::Tensor dadvantage_dev = advantage_dev >= 0.0F;
    t::Tensor importance_dev = t::div(avs_dev.actor_out, oad_dev.detach() + 1E-7) * advantage_dev.detach();
    t::Tensor clip_dev = (ones_dev.squeeze() + mp.epsilon) * (advantage_dev * dadvantage_dev) +
                         (ones_dev.squeeze() - mp.epsilon) * (advantage_dev * dadvantage_dev.logical_not());
    t::Tensor dmin_dev = importance_dev < clip_dev;
    t::Tensor ppo_loss_dev = t::mean(importance_dev * dmin_dev.detach() + clip_dev.detach() * dmin_dev.logical_not().detach());

    //V loss
    t::Tensor value_loss_dev = t::mse_loss(avs_dev.critic_out, target_value_dev.detach());

    //PPO loss + V loss + ICM F loss + ICM I loss
    t::Tensor loss_dev = ppo_loss_dev + value_loss_dev + icmf_loss_dev + icmi_loss_dev;

    loss_dev.backward();
    rlm.optimizer.step();

    if (i % loss_sampling_interval == 0){
      s::cout << "forward dynamics loss: " << icmf_loss_dev.item().to<float>() << s::endl;
      s::cout << "inverse dynamics loss: " << icmi_loss_dev.item().to<float>() << s::endl;
      s::cout << "PPO Loss: " << ppo_loss_dev.item().to<float>() << s::endl;
      s::cout << "Critic Loss: " << value_loss_dev.item().to<float>() << s::endl;
      s::cout << "Loss: " << loss_dev.item().to<float>() << s::endl;
      losses.push_back(loss_dev.item().to<float>());
    }
  }
}

} //gridworld_pt

#endif//GRIDWORLD_PPO_LEARNING
