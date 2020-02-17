#include <cassert>
#include <ctime>
#include <cstdlib>
#include <cctype>
#include <string>
#include <random>
#include <chrono>
#include <vector>
#include <iostream>
#include <filesystem>
#include <algorithm>

#include <type_alias.h>
#include <types.h>
#include <go_types.h>
#include <splitmix.h>
#include <dirichlet_distribution.h>
#include <pytorch_util.h>
#include <experience/zero_episodic_buffer.h>
#include <encoders/go_action_encoder.h>
#include <encoders/go_zero_encoder.h>
#include <models/model_base.h>
#include <models/zero_model_small.h>
#include <agents/zero_agent.h>

#include <torch/torch.h>

namespace s = std;
namespace c = s::chrono;
namespace t = torch;
namespace R = rlgames;

constexpr ubyte SZ = 9;

int main(int argc, const char* argv[]){
  uint episodes = 1000;
  uint batchsize = 10;
  s::string model_config_file;
  s::string model_file;
  s::string optimizer_file;
  s::string result_file;

  if (argc != 7){
    s::cout << "Usage: zero_small_selfplay <episodes> <batchsize> <model_config> <model_file> <optimizer_file> <result_file>" << s::endl;
    s::exit(1);
  }

  episodes  = atoi(argv[1]);
  batchsize = atoi(argv[2]);
  model_config_file = argv[3];
  model_file = argv[4];
  optimizer_file = argv[5];
  result_file = argv[6];

  if (not s::filesystem::exists(model_config_file)){
    s::cout << "model configuration file does not exist" << s::endl;
    s::exit(1);
  }

  srand(time(nullptr));
  uint max_bsize = batchsize * SZ * SZ * SZ;

  t::Device device(t::kCPU);
  if (t::cuda::is_available()){
    s::cout << "Using GPU" << s::endl;
    device = t::Device(t::kCUDA);
  }

  R::ZeroGoStateEncoder<SZ> state_encoder;
  R::ZeroGoActionEncoder<SZ> action_encoder;

  R::TensorDimP state_size = state_encoder.state_size();
  constexpr uint action_size = R::ZeroGoActionEncoder<SZ>::action_size();

  R::ModelContainer<R::ZeroModelSmall, R::ZeroGoStateEncoder<SZ>, R::ZeroGoActionEncoder<SZ>, t::optim::Adam> model_container(
    R::ZeroModelSmall(state_size, action_size, R::load_model_option<R::ZeroModelSmallOptions>(model_config_file)),
    s::move(state_encoder),
    s::move(action_encoder),
    1E-5F //learning_rate
  );

  if (s::filesystem::exists(model_file) && s::filesystem::exists(optimizer_file)){
    s::cout << "Using saved model and optimizer parameters" << s::endl;
    R::load_model(model_container, model_file, optimizer_file);
  }

  model_container.model->to(device);

  //the agents will share the same model, but use a different experience collector buffer
  R::ZeroAgent<decltype(model_container), R::dirichlet_distribution<action_size>, R::Splitmix, R::GoBoard<SZ>, R::GoGameState<SZ>, action_size> agent1(
    model_container,
    device,
    1600,   /*max expansion*/
    0.2,    /*exploration factor*/
    0.3,    /*dirichlet distribution alpha*/
    rand()  /*random seed*/
  );

  R::ZeroAgent<decltype(model_container), R::dirichlet_distribution<action_size>, R::Splitmix, R::GoBoard<SZ>, R::GoGameState<SZ>, action_size> agent2(
    model_container,
    device,
    1600,   /*max expansion*/
    0.2,    /*exploration factor*/
    0.3,    /*dirichlet distribution alpha*/
    rand()  /*random seed*/
  );

  s::vector<float> losses;
  uint64 a1_wins = 0, a2_wins = 0, tie_count = 0;

  uint reporting_interval = s::max(episodes / 10U, 1U);

  for (uint i = 0; i < episodes; ++i){
    R::ZeroEpisodicExpCollector buffer1(max_bsize, state_size, action_size, device);
    R::ZeroEpisodicExpCollector buffer2(max_bsize, state_size, action_size, device);
    agent1.set_exp(buffer1);
    agent2.set_exp(buffer2);

    R::ZeroExperience experience(device);
    for (uint j = 0; j < batchsize; ++j){
      R::GoGameState<SZ> state;
      R::Player turn = R::Player::Black;

      auto gstart = c::high_resolution_clock::now();

      while (not state.is_over()){
        R::Move move(R::M::Pass);
        switch (turn){
        case R::Player::Black: move = agent1.select_move(state); break;
        case R::Player::White: move = agent2.select_move(state); break;
        default: assert(false);
        }
        state.apply_move(move);
        turn = R::other_player(turn);
      }

      R::Player winner = state.winner();
      switch (winner){
      case R::Player::Black:
        buffer1.complete_episode(decltype(agent1)::MAX_SCORE);
        buffer2.complete_episode(decltype(agent2)::MIN_SCORE);
        a1_wins += 1;
        break;
      case R::Player::White:
        buffer1.complete_episode(decltype(agent1)::MIN_SCORE);
        buffer2.complete_episode(decltype(agent2)::MAX_SCORE);
        a2_wins += 1;
        break;
      case R::Player::Unknown:
        buffer1.complete_episode(decltype(agent1)::TIE_SCORE);
        buffer2.complete_episode(decltype(agent2)::TIE_SCORE);
        tie_count += 1;
        break;
      default: assert(false);
      }

      auto gstop = c::high_resolution_clock::now();
      auto duration = c::duration_cast<c::microseconds>(gstop - gstart);

      s::cout << "Game time: " << duration.count() << " microseconds" << s::endl;

      R::append_experiences(experience, buffer1, buffer2);
    }
    float loss = train(model_container, experience);

    if (i % reporting_interval){
      s::cout << "Episode " << i << ". Loss " << loss << s::endl;
    }

    losses.push_back(loss);
  }

  s::cout << "Self play complete. Saving training and result." << s::endl;

  R::save_model(model_container, model_file, optimizer_file);

  R::save_training_result(result_file, losses, a1_wins, a2_wins, tie_count);
}
