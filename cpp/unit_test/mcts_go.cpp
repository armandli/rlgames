#include <cassert>
#include <cctype>

#include <chrono>
#include <iostream>

#include <type_alias.h>
#include <types.h>
#include <go_types.h>
#include <splitmix.h>
#include <mcts_agent.h>
#include <parallel_mcts_agent.h>

namespace s = std;
namespace c = s::chrono;
namespace R = rlgames;

constexpr ubyte SZ = 9;

int main(){
  R::GoGameState<SZ> state;
  //R::MCTSAgent<R::Splitmix, R::GoBoard<SZ>, R::GoGameState<SZ>> agent(531441, 1., 1);
  //R::MCTSAgent<R::Splitmix, R::GoBoard<SZ>, R::GoGameState<SZ>> agent(531441, 1., 64);
  R::PMCTSAgent<R::Splitmix, R::GoBoard<SZ>, R::GoGameState<SZ>> agent(531441, 1., 128);
  R::Player turn = R::Player::Black;

  while (not state.is_over()){
    s::cout << state.board() << s::endl;

    R::Move move(R::M::Pass);

    auto tstart = c::high_resolution_clock::now();

    switch (turn){
    case R::Player::Black: move = agent.select_move(state); break;
    case R::Player::White: move = agent.select_move(state); break;
    default: assert(false);
    }

    auto tstop = c::high_resolution_clock::now();
    auto duration = c::duration_cast<c::microseconds>(tstop - tstart);

    s::cout << duration.count() << " microseconds" << s::endl;

    state.apply_move(move);
    turn = R::other_player(turn);
  }
  s::cout << state.board() << s::endl;
  R::Player winner = state.winner();
  switch (winner){
  case R::Player::Black: case R::Player::White:
    s::cout << winner << " won";
    break;
  case R::Player::Unknown:
    s::cout << "ties";
    break;
  default: assert(false);
  }
  s::cout << s::endl;
}
