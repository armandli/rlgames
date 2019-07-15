#include <unistd.h>
#include <iostream>

#include <types.h>
#include <ttt_types.h>
#include <minimax_agent.h>

namespace s = std;
namespace R = rlgames;

int main(){
  R::TTTGameState state;
  R::MinimaxAgent<R::TTTBoard, R::TTTGameState, float(const R::TTTBoard&), true> agent(5, R::nil_evaluator<R::TTTBoard>);

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
