#ifndef GRIDWORLD_NAIVE_QLEARNING
#define GRIDWORLD_NAIVE_QLEARNING

#include <random>

#include <torch/torch.h>

namespace gridworld_pt {

namespace s = std;
namespace t = torch;

//Naive Q Learning with epsilon greedy exploration
//Optimization: Q function returns Q value for all actions of the same state
//instead of taking state and action as input and predict Q value

//TODO: make use of GPU only
//TODO: use checkpoints

template <typename ENV, typename RLM, typename INS, typename ACTION>
void naive_qlearning(ENV& env, RLM& rlm, uint64 epochs, double gamma, double epsilon, s::vector<float>& losses, uint64 max_steps){
  s::uniform_real_distribution<double> dist(0., 1.);
  s::uniform_int_distribution<uint> rand_action(0U, env.action_size() - 1);
  s::default_random_engine reng;

  for (uint64 i = 0; i < epochs; ++i){
    INS ins = env.create();
    uint64 step_count = 0;
    while (not env.is_termination(ins) && step_count < max_steps){
      rlm.optimizer.zero_grad();
      t::Tensor tstate = rlm.state_encoder.encode_state(env.get_state(ins));
      t::Tensor qval = rlm.model->forward(tstate);
      ACTION action;
      // epsilon greedy action selection
      if (dist(reng) < epsilon){
        action = (ACTION)rand_action(reng);
      } else {
        action = rlm.action_encoder.decode_action(qval);
      }
      env.apply_action(ins, action);
      t::Tensor tnstate = rlm.state_encoder.encode_state(env.get_state(ins));
      float reward = env.get_reward(ins);
      t::Tensor nqval = rlm.model->forward(tnstate);
      float maxq = nqval.max().item().to<float>();
      t::Tensor y = qval.clone();
      if (env.is_termination(ins)){
        y[(uint)action] = t::scalar_tensor(reward);
      } else {
        y[(uint)action] = t::scalar_tensor(reward + gamma * maxq);
      }
      t::Tensor loss = t::mse_loss(qval, y.detach());
      loss.backward();
      losses.push_back(loss.item().to<float>());
      rlm.optimizer.step();

      step_count++;
    }

    // epsilon decay
    if (epsilon > 0.1)
      epsilon -= 1. / (double)epochs;

    if (i % 10 == 0)
      s::cout << "epsilon: " << epsilon << s::endl;
  }
}

} //gridworld_pt

#endif//GRIDWORLD_NAIVE_QLEARNING
