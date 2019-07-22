#include <cassert>
#include <cctype>

#include <iostream>

#include <type_alias.h>
#include <types.h>
#include <go_types.h>
#include <mcts_agent.h>

namespace s = std;
namespace R = rlgames;

constexpr ubyte SZ = 9;

R::Move parse_human_move(){
  R::Move ret(R::M::Pass);
  s::string input;

  do {
    s::cout << "-- ";
    s::getline(s::cin, input);
    if (input.size() == 0) // pass
      break;
    if (input.size() < 2){
      s::cout << "Invalid human input. retrying." << s::endl;
      continue;
    }
    char      colstr = input[0];
    s::string rowstr = input.substr(1);
    int colidx, rowidx;
    if (s::isupper(colstr))
      colidx = colstr - 'A';
    else
      colidx = colstr - 'a';
    try {
      rowidx = s::stoi(rowstr);
    } catch (s::invalid_argument& err){
      s::cout << "Invalid row number: " << rowstr << ". Need to be a positive integer. retrying." << s::endl;
      continue;
    }
    if (colidx < 0 || colidx >= SZ){
      s::cout << "Invalid col number: " << colstr << ". retrying." << s::endl;
      continue;
    }
    if (rowidx < 0 || rowidx >= SZ){
      s::cout << "Invalid row number: " << rowstr << ". retrying." << s::endl;
      continue;
    }
    ret = R::Move(R::M::Play, R::Pt(rowidx, colidx));
    break;
  } while (true);

  return ret;
}

int main(){
  R::GoGameState<SZ> state;
  R::MCTSAgent<R::GoBoard<SZ>, R::GoGameState<SZ>> agent(7000, 1., 32);
  R::Player turn = R::Player::Black;

  while (not state.is_over()){
    s::cout << state.board() << s::endl;

    R::Move move(R::M::Pass);
    switch (turn){
    case R::Player::Black: move = parse_human_move(); break;
    case R::Player::White: move = agent.select_move(state); break;
    default: assert(false);
    }
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
