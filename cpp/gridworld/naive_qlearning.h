#ifndef GRIDWORLD_NAIVE_QLEARNING
#define GRIDWORLD_NAIVE_QLEARNING

#include <learning_metaparam.h>

#include <random>

#include <torch/torch.h>

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

//Naive Q Learning with epsilon greedy exploration
//Optimization: Q function returns Q value for all actions of the same state
//instead of taking state and action as input and predict Q value

template <typename ENV, typename RLM, typename INS, typename ACTION, uint loss_sampling_interval = 100U>
void naive_qlearning(
  ENV& env,
  RLM& rlm,
  t::Device device,
  qlearning_metaparams<epsilon_greedy_metaparams, experience_replay_metaparams> mp,
  s::vector<float>& losses){

  t::Device cpu_device(t::kCPU);

  s::uniform_real_distribution<double> dist(0., 1.);
  s::uniform_int_distribution<uint> rand_action(0U, env.action_size() - 1);
  s::default_random_engine reng;

  rlm.model->to(device);

  for (uint64 i = 0; i < mp.epochs; ++i){
    INS ins = env.create();
    uint64 step_count = 0;
    while (not env.is_termination(ins) && step_count < mp.max_steps){
      rlm.model->zero_grad();
      t::Tensor tstate_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
      t::Tensor qval_dev = rlm.model->forward(tstate_dev);
      ACTION action;
      // epsilon greedy action selection
      if (dist(reng) < mp.exp.epsilon){
        action = (ACTION)rand_action(reng);
      } else {
        action = rlm.action_encoder.decode_action(qval_dev);
      }
      env.apply_action(ins, action);
      t::Tensor tnstate_dev = rlm.state_encoder.encode_state(env.get_state(ins), device);
      float reward = env.get_reward(ins);
      t::Tensor nqval_dev = rlm.model->forward(tnstate_dev);
      float maxq = nqval_dev.max().item().to<float>();
      t::Tensor y_dev = qval_dev.clone();
      if (env.is_termination(ins)){
        y_dev[(uint)action] = t::scalar_tensor(reward, device);
      } else {
        y_dev[(uint)action] = t::scalar_tensor(reward + mp.gamma * maxq, device);
      }
      t::Tensor loss_dev = t::mse_loss(qval_dev, y_dev.detach());
      loss_dev.backward();
      rlm.optimizer.step();

      if ((i + step_count) % loss_sampling_interval == 0)
        losses.push_back(loss_dev.item().to<float>());

      step_count++;
    }

    // epsilon decay
    if (mp.exp.epsilon > 0.1)
      mp.exp.epsilon -= 1. / (double)mp.epochs;

    if (i % 10 == 0)
      s::cout << "epsilon: " << mp.exp.epsilon << s::endl;
  }
}

} //gridworld_pt

#endif//GRIDWORLD_NAIVE_QLEARNING
