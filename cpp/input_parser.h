#ifndef RLGAMES_INPUT_PARSER
#define RLGAMES_INPUT_PARSER

#include <string>
#include <optional>

#include <type_alias.h>
#include <types.h>

namespace rlgames {

namespace s = std;

s::optional<Move> parse_board_index(const s::string& input, int board_size){
  if (input.size() < 2) return {};
  char colstr = input[0];
  s::string rowstr = input.substr(1);
  int colidx, rowidx;
  if (s::isupper(colstr)){
    if (colstr == 'I') return {};
    colidx = colstr - 'A';
    if (colstr >= 'J')
      colidx -= 1;
  } else {
    if (colstr == 'i') return {};
    colidx = colstr - 'a';
    if (colstr >= 'j')
      colidx -= 1;
  }
  try {
    rowidx = s::stoi(rowstr) - 1;
  } catch (s::invalid_argument& err){
    return {};
  }
  if (colidx < 0 || colidx >= board_size) return {};
  if (rowidx < 0 || rowidx >= board_size) return {};
  return s::make_optional<Move>(M::Play, Pt(rowidx, colidx));
}

s::optional<Move> get_board_index_from_stdio(uint board_size){
  s::string input;
  s::getline(s::cin, input);
  return parse_board_index(input, board_size);
}

} // rlgames

#endif//RLGAMES_INPUT_PARSER
