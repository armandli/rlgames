#ifndef GRIDWORLD_AC_LEARNING
#define GRIDWORLD_AC_LEARNING

#include <learning_util.h>
#include <learning_debug.h>
#include <learning_metaparam.h>
#include <experience_buffer.h>

#include <vector>
#include <random>

#include <torch/torch.h>

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

//Batch advantage Actor Critic Implementation
//uses Experience Buffer as batch
//uses target function

template <typename ENV, typename RLM, typename INS, typename ACTION, uint loss_sampling_interval = 100>
void ac_learning(
  ENV& env,
  RLM& rlm,
  t::Device device,
  ac_metaparams& mp,
  s::vector<float>& losses,
  uint64 random_seed){

  t::Device cpu_device(t::kCPU);
  s::default_random_engine reng(random_seed);
  ExpBuffer<ACTION> buffer(mp.batchsize * mp.max_steps, env.state_size(), device);

  decltype(rlm.model) targetn(rlm.model);
  rlm.model->to(device);
  targetn->to(device);

  for (uint64 i = 0, tc_step = 0; i < mp.epochs; ++i){
    buffer.reset();
    for (uint64 b = 0; b < mp.batchsize; ++b){
      INS ins = env.create();
      uint step_count = 0;
      while (not env.is_termination(ins) && step_count < mp.max_steps){
        if (tc_step >= mp.tc_steps){
          copy_state(targetn, rlm.model);
          tc_step = 0;
        }
        t::Tensor tstate_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
        t::Tensor tadist_dev = rlm.model->actor_forward(tstate_dev);
        t::Tensor tadist = tadist_dev.to(cpu_device);
        ACTION action = (ACTION)sample_discrete_distribution(tadist.data_ptr<float>(), env.action_size(), reng);
        env.apply_action(ins, action);
        float reward = env.get_reward(ins);
        t::Tensor tnstate_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
        buffer.append(Exp<ACTION>(tstate_dev, action, reward, tnstate_dev, env.is_termination(ins)));

        step_count++;
        tc_step++;
      }
    }

    rlm.model->zero_grad();

    ARTArray actions_rewards = buffer.actions_n_rewards();
    t::Tensor action = t::from_blob(actions_rewards.actions.data(), {(sint64)actions_rewards.actions.size()}, t::kLong);
    t::Tensor action_dev = action.to(device);
    t::Tensor reward = t::from_blob(actions_rewards.rewards.data(), {(sint64)actions_rewards.rewards.size()});
    t::Tensor reward_dev = reward.to(device);
    t::Tensor terminal = t::from_blob(actions_rewards.is_terminals.data(), {(sint64)actions_rewards.is_terminals.size()}, t::kLong);
    t::Tensor terminal_dev = terminal.to(device);
    terminal_dev = terminal_dev.logical_not();

    ACTensor avs = rlm.model->forward(buffer.states_tensor());
    t::Tensor nvs = targetn->critic_forward(buffer.nstates_tensor());

    avs.actor_out = t::index_select(avs.actor_out, 1, action_dev);
    avs.actor_out = avs.actor_out.diagonal();

    //the advantage function r + gamma * V(s') - V(s)
    t::Tensor g = reward_dev.detach() + mp.gamma * terminal_dev.detach() * nvs.detach() - avs.critic_out;
    t::Tensor loss_dev = t::mean(-1.F * g * t::log(avs.actor_out));

    loss_dev.backward();
    rlm.optimizer.step();

    if (i % loss_sampling_interval == 0){
      s::cout << "Loss: " << loss_dev.item().to<float>() << s::endl;
      losses.push_back(loss_dev.item().to<float>());
    }
  }
}

} // gridworld_pt

#endif//GRIDWORLD_AC_LEARNING
