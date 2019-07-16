#include <gtest/gtest.h>

#include <boost/spirit/include/qi.hpp>

#include <types.h>
#include <sgf.h>

namespace s = std;
namespace f = sgf;
namespace r = rlgames;
namespace qi = boost::spirit::qi;

struct MockSGFFileReader : public f::SGFFileReader {
  using f::SGFFileReader::RawData;
  using f::SGFFileReader::SGFGrammar;
  using f::SGFFileReader::spirit_parse;
  using f::SGFFileReader::read_file;
  using f::SGFFileReader::read_metadata;
  using f::SGFFileReader::read_moves;
  using f::SGFFileReader::mstr_to_pt;
};

struct TestSGFReader : ::testing::Test {
  TestSGFReader(){}
  ~TestSGFReader(){}

  MockSGFFileReader reader;
};

TEST_F(TestSGFReader, TestReadFile1){
  s::string data = reader.read_file("test.sgf");
  EXPECT_TRUE(data.size() > 0);
}

TEST_F(TestSGFReader, TestRawParse1){
  s::string data = reader.read_file("test.sgf");
  MockSGFFileReader::RawData raw_content;
  MockSGFFileReader::SGFGrammar grammar;

  reader.spirit_parse(data, grammar, qi::space, raw_content);
  EXPECT_TRUE(raw_content.size() > 1);
}

TEST_F(TestSGFReader, TestRead1){
  f::SGFData data = reader.parse_sgf_file("test.sgf");
  EXPECT_EQ(1, data.metadata.game_number);
  EXPECT_EQ(4, data.metadata.ff_version);
  EXPECT_EQ(19, data.metadata.board_size);
  EXPECT_EQ(0.5, data.metadata.komi);
  EXPECT_STREQ("W+Resign", data.metadata.result.c_str());
  EXPECT_STREQ("Chinese", data.metadata.rule.c_str());
  ASSERT_EQ(154, data.moves.size());
  EXPECT_EQ(r::Player::Black, data.moves[0].player);
  EXPECT_EQ(r::Player::White, data.moves[1].player);
  EXPECT_EQ(3, data.moves[0].move.mpt.c);
  EXPECT_EQ(10, data.moves[153].move.mpt.c);
  EXPECT_EQ(12, data.moves[153].move.mpt.r);
}
