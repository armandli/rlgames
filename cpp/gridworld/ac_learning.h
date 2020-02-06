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
  ExpBuffer<ACTION> buffer(mp.batchsize * mp.max_steps, rlm.state_encoder.state_size().flatten_size(), device);

  decltype(rlm.model) targetn(rlm.model);
  copy_state(targetn, rlm.model);

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
        ACTION action = (ACTION)sample_discrete_distribution(tadist.data_ptr<float>(), rlm.action_encoder.action_size(), reng);
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
    t::Tensor action_dev = t::from_blob(actions_rewards.actions.data(), {(sint64)actions_rewards.actions.size()}, t::kLong).to(device);
    t::Tensor reward_dev = t::from_blob(actions_rewards.rewards.data(), {(sint64)actions_rewards.rewards.size()}).to(device);
    t::Tensor terminal_dev = t::from_blob(actions_rewards.is_terminals.data(), {(sint64)actions_rewards.is_terminals.size()}, t::kLong).to(device).logical_not();

    ACTensor avs_dev = rlm.model->forward(buffer.states_tensor());
    avs_dev.actor_out = t::index_select(avs_dev.actor_out, 1, action_dev).diagonal();
    t::Tensor nvs_dev = targetn->critic_forward(buffer.nstates_tensor());

    //the advantage function r + gamma * V(s') - V(s)
    t::Tensor targetv_dev = reward_dev + mp.gamma * terminal_dev * nvs_dev;
    t::Tensor g = targetv_dev - avs_dev.critic_out;

    //loss = actor loss + value function loss
    t::Tensor actor_loss_dev = t::mean(-1.F * g.detach() * t::log(avs_dev.actor_out));
    t::Tensor critic_loss_dev = t::mse_loss(avs_dev.critic_out, targetv_dev.detach());
    t::Tensor loss_dev = actor_loss_dev + critic_loss_dev;

    loss_dev.backward();
    rlm.optimizer.step();

    if (i % loss_sampling_interval == 0){
      s::cout << "Actor Loss: " << actor_loss_dev.item().to<float>() << s::endl;
      s::cout << "Critic Loss: " << critic_loss_dev.item().to<float>() << s::endl;
      s::cout << "Loss: " << loss_dev.item().to<float>() << s::endl;
      losses.push_back(loss_dev.item().to<float>());
    }
  }
}

} // gridworld_pt

#endif//GRIDWORLD_AC_LEARNING
