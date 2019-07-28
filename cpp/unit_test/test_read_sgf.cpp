#include <iostream>

#include <boost/spirit/include/qi.hpp>

#include <types.h>
#include <sgf.h>
#include <go_types.h>

namespace s = std;
namespace f = sgf;
namespace r = rlgames;
namespace qi = boost::spirit::qi;

int main(int argc, char *argv[]){
  if (argc != 2){
    s::cout << "Usage: " << argv[0] << " sgf_file" << s::endl;
    exit(1);
  }

  f::SGFFileReader reader;
  f::SGFData data = reader.parse_sgf_file(argv[1]);

  r::GoGameState<19> gs;

  s::cout << gs.board() << s::endl;
  for (r::PlayerMove mv : data.moves){
    gs.apply_move(mv.move);
    s::cout << gs.board() << s::endl;
    s::cout << mv.move << s::endl;
  }
}
