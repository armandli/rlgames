#ifndef RLGAMES_INPUT_PARSER
#define RLGAMES_INPUT_PARSER

#include <string>
#include <iostream>

#include <type_alias.h>
#include <types.h>

namespace rlgames {

namespace s = std;

Move parse_move(uint board_size){
  s::string input;
  do {
    s::getline(s::cin, input);
    Move m = string_to_move(input, board_size);
    if (m.mty != M::Unknown) return m;
    s::cout << "unable to parse input. try again" << s::endl;
  } while (true);
}

} // rlgames

#endif//RLGAMES_INPUT_PARSER
