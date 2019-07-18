#include <unistd.h>
#include <iostream>

#include <types.h>
#include <go_types.h>
#include <random_agent.h>

namespace s = std;
namespace R = rlgames;

constexpr ubyte SZ = 19;

int main(){
  R::GoGameState<SZ> state;
  R::RandomAgent<R::GoBoard<SZ>, R::GoGameState<SZ>> agent;

  while (not state.is_over()){
    usleep(1000);

    s::cout << state.board() << s::endl;
    R::Move move = agent.select_move(state);
    s::cout << state.next_player() << " " << move << s::endl;
    state.apply_move(move);
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
