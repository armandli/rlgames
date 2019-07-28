#ifndef RLGAMES_SGF
#define RLGAMES_SGF

#include <cassert>
#include <cerrno>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix.hpp>

#include <type_alias.h>
#include <types.h>

namespace s = std;
namespace qi = boost::spirit::qi;
namespace phx = boost::phoenix;
namespace r = rlgames;

namespace sgf {

struct SGFMetadata {
  uint      ff_version;         //FF[K]
  uint      game_number;        //GM[K]
  uint      board_size;         //SZ[K]
  uint      handicap;           //HA[K]
  float     komi;               //KM[F]
  s::string rule;               //RU[S]
  s::string result;             //RE[Player+F]
};

struct SGFData {
  SGFMetadata              metadata;
  s::vector<r::PlayerMove> moves;    //W[ab] for white move Pt(0, 1) and B[ef] for black move Pt(5, 6)
};

class SGFFileReader {
protected:
  using RawData = s::vector<s::vector<s::vector<s::string>>>;

  struct SGFGrammar: public qi::grammar<s::string::const_iterator, RawData(), qi::space_type> {
    using Iterator = s::string::const_iterator;

    SGFGrammar(): SGFGrammar::base_type(start) {
      prop %= qi::alpha >> *(~qi::char_('['));
      prop_content %= *(~qi::char_(']'));
      clause %= prop >> '[' >> prop_content >> ']' >> *('[' >> prop_content >> ']');
      segment %= ';' >> clause >> *clause;
      start = '(' >> segment >> *(segment) >> ')';
    }

    qi::rule<Iterator, s::string()> prop;
    qi::rule<Iterator, s::string()> prop_content;
    qi::rule<Iterator, s::vector<s::string>(), qi::space_type> clause;
    qi::rule<Iterator, s::vector<s::vector<s::string>>(), qi::space_type> segment;
    qi::rule<Iterator, RawData(), qi::space_type> start;
  };

  template <typename Parser, typename Skipper, typename... Args>
  void spirit_parse(const s::string& input, const Parser& p, Skipper& skip, Args&&... args){
    s::string::const_iterator begin = input.begin(), end = input.end();
    qi::phrase_parse(begin, end, p, skip, s::forward<Args>(args)...);
    if (begin != end){
      s::cout << "Unparseable: " << s::quoted(s::string(begin, end)) << s::endl;
      throw s::runtime_error("Parse Error!");
    }
  }

  s::string read_file(const char* filename){
    s::ifstream in(filename, s::ios::in | s::ios::binary);
    if (in){
      s::string content;
      in.seekg(0, s::ios::end);
      content.resize(in.tellg());
      in.seekg(0, s::ios::beg);
      in.read(&content[0], content.size());
      in.close();
      return content;
    }
    throw errno;
  }

  void read_metadata(SGFData& ret, const RawData& content){
    for (const s::vector<s::string>& clause : content[0]){
      if (clause.size() == 0) continue;

      if (clause[0] == "FF"){
        if (clause.size() == 2){
          try {
            uint version = s::stoul(clause[1]);
            ret.metadata.ff_version = version;
          } catch (const s::invalid_argument& err){
            s::cerr << "Invalid FF version number " << clause[1] << ". Version ignored" << s::endl;
            s::cerr << err.what() << s::endl;
          }
        }
      } else if (clause[0] == "GM"){
        if (clause.size() == 2){
          try {
            uint game_number = s::stoul(clause[1]);
            ret.metadata.game_number = game_number;
          } catch (const s::invalid_argument& err){
            s::cerr << "Invalid GM number " << clause[1] << ". Number ignored" << s::endl;
            s::cerr << err.what() << s::endl;
          }
        }
      } else if (clause[0] == "SZ"){
        if (clause.size() == 2){
          try {
            uint board_size = s::stoul(clause[1]);
            ret.metadata.board_size = board_size;
          } catch (const s::invalid_argument& err){
            s::cerr << "Invalid SZ size " << clause[1] << ". Number ignored" << s::endl;
            s::cerr << err.what() << s::endl;
          }
        }
      } else if (clause[0] == "HA"){
        if (clause.size() == 2){
          try {
            uint handicap = s::stoul(clause[1]);
            ret.metadata.handicap = handicap;
          } catch (const s::invalid_argument& err){
            s::cerr << "Invalid HA handicap number " << clause[1] << ". Number ignored" << s::endl;
            s::cerr << err.what() << s::endl;
          }
        }

      } else if (clause[0] == "KM"){
        if (clause.size() == 2){
          try {
            float komi = s::stof(clause[1]);
            ret.metadata.komi = komi;
          } catch (const s::invalid_argument& err){
            s::cerr << "Invalid KM komi value " << clause[1] << ". Number ignored" << s::endl;
            s::cerr << err.what() << s::endl;
          }
        }
      } else if (clause[0] == "RU"){
        if (clause.size() == 2)
          ret.metadata.rule = clause[1];
      } else if (clause[0] == "RE"){
        if (clause.size() == 2)
          ret.metadata.result = clause[1];
      }
    }
  }

  r::Move mstr_to_pt(const s::string& mstr){
    //TODO: make sure first character represent row and second represent column
    return r::Move(r::M::Play, r::Pt(mstr[0] - 'a', mstr[1] - 'a'));
  }

  void read_moves(SGFData& ret, const RawData& content){
    for (size_t i = 1; i < content.size(); ++i){
      for (const s::vector<s::string>& clause : content[i]){
        if (clause.size() == 0) continue;

        if (clause[0] == "B"){
          if (clause[1].size() == 0){
            ret.moves.push_back(r::PlayerMove(r::Player::Black, r::Move(r::M::Pass)));
            continue;
          }
          if (clause[1].size() != 2){
            s::cerr << "Unexpected move description: " << clause[1] << ". move ignored." << s::endl;
            continue;
          }
          ret.moves.push_back(r::PlayerMove(r::Player::Black, mstr_to_pt(clause[1])));
        } else if (clause[0] == "W"){
          if (clause[1].size() == 0){
            ret.moves.push_back(r::PlayerMove(r::Player::White, r::Move(r::M::Pass)));
            continue;
          }
          if (clause[1].size() != 2){
            s::cerr << "Unexpected move description: " << clause[1] << ". move ignored." << s::endl;
            continue;
          }
          ret.moves.push_back(r::PlayerMove(r::Player::White, mstr_to_pt(clause[1])));
        }
      }
    }
  }
public:
  SGFData parse_sgf_file(const char* filename){
    SGFData ret;
    s::string raw_data = read_file(filename);
    RawData raw_content;
    SGFGrammar grammar;
    spirit_parse(raw_data, grammar, qi::space, raw_content);

    if (raw_content.size() < 1U){
      s::cerr << "ERROR: cannot obtain any segment from file " << filename << s::endl;
      return ret;
    }

    read_metadata(ret, raw_content);
    read_moves(ret, raw_content);
    return ret;
  }
};

} //sgf

#endif//RLGAMES_SGF
