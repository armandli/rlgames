#ifndef RLGAMES_MINIMAX_AGENT
#define RLGAMES_MINIMAX_AGENT

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <random>

#include <type_alias.h>
#include <types.h>
#include <agent_base.h>

namespace rlgames {

template <typename Board>
float nil_evaluator(const Board&){
  return 0.F;
}

template <typename Board, typename GameState, typename EvalFn, bool ABP = true>
struct MinimaxAgent : AgentBase<Board, GameState, MinimaxAgent<Board, GameState, EvalFn, ABP>> {
  static constexpr float MIN_SCORE = -1.;
  static constexpr float MAX_SCORE =  1.;
  static constexpr float TIE_SCORE =  0.;
private:
  size_t mDepth;
  EvalFn& mEval;
protected:
  struct RRet {
    float score;
    Move  move;

    RRet() = default;
    RRet(float score, Move move): score(score), move(move) {}
    RRet(const RRet& o): score(o.score), move(o.move) {}
    RRet& operator=(const RRet& o){
      score = o.score;
      move = o.move;
      return *this;
    }
  };

  bool is_improvement(float score, float best_score, Player player){
    switch (player){
    case Player::Black: return score > best_score;
    case Player::White: return score < best_score;
    default: assert(false);
    }
  }

  //TODO: make sure recursion does not build up stack
  RRet recursive_minimax_search(const GameState& gs, size_t depth){
    if (gs.is_over()){
      Player winner = gs.winner();
      switch (winner){
      case Player::Black:   return RRet(MAX_SCORE, Move(M::Unknown));
      case Player::White:   return RRet(MIN_SCORE, Move(M::Unknown));
      case Player::Unknown: return RRet(TIE_SCORE, Move(M::Unknown));
      default: assert(false);
      }
    }
    if (depth == 0)
      return RRet(mEval(gs.board()), Move(M::Unknown));
    s::vector<Move> best_moves;
    float best_score;
    switch (gs.next_player()){
    case Player::Black: best_score = MIN_SCORE; break;
    case Player::White: best_score = MAX_SCORE; break;
    default: assert(false);
    }
    for (Move m : gs.legal_moves()){
      GameState ngs = gs;
      ngs.apply_move(m);
      size_t ndepth = depth;
      if (ngs.next_player() == Player::Black)
        ndepth--;
      RRet val = recursive_minimax_search(ngs, ndepth);
      if (is_improvement(val.score, best_score, gs.next_player())){
        best_score = val.score;
        best_moves.clear();
        best_moves.push_back(m);
      } else if (val.score == best_score)
        best_moves.push_back(m);
    }
    if (best_moves.size() > 0){
      uint choice = s::rand() % best_moves.size();
      return RRet(best_score, best_moves[choice]);
    } else
      return RRet(best_score, Move(M::Pass));
  }

  bool should_prune(float score, float alpha, float beta, Player player){
    switch (player){
    case Player::Black: return score > beta;
    case Player::White: return score < alpha;
    default: assert(false);
    }
  }

  //TODO: make sure recursion does not build up stack
  /* Alpha-Beta Pruning: introduce new parameters alpha and beta.
   * alpha is the best value available to the maximizer from the parent to the root
   * beta is the best value available to the minimizer from the parent to the root
   * we use those to prune segments downstream.
   * if the current node is a maximizer, and the beta it has received from its
   * parent (minimizer) has lower value than maximum value found by current
   * node, we no longer need to explore downstream.
   * if the current node is the minimizer, and the alpha it has received from
   * its parent (maximizer) has higher value than the minimum value found
   * current node, we no longer need to explore downstream.
   */
  RRet recursive_alpha_beta_minimax_search(const GameState& gs, size_t depth, float alpha, float beta){
    if (gs.is_over()){
      Player winner = gs.winner();
      switch (winner){
      case Player::Black:   return RRet(MAX_SCORE, Move(M::Unknown));
      case Player::White:   return RRet(MIN_SCORE, Move(M::Unknown));
      case Player::Unknown: return RRet(TIE_SCORE, Move(M::Unknown));
      default: assert(false);
      }
    }
    if (depth == 0)
      return RRet(mEval(gs.board()), Move(M::Unknown));
    s::vector<Move> best_moves;
    float best_score;
    switch (gs.next_player()){
    case Player::Black: best_score = MIN_SCORE; break;
    case Player::White: best_score = MAX_SCORE; break;
    default: assert(false);
    }
    for (Move m : gs.legal_moves()){
      if (should_prune(best_score, alpha, beta, gs.next_player()))
        break;
      GameState ngs = gs;
      ngs.apply_move(m);
      size_t ndepth = depth;
      if (ngs.next_player() == Player::Black)
        ndepth--;
      RRet val = recursive_alpha_beta_minimax_search(ngs, ndepth, alpha, beta);
      if (is_improvement(val.score, best_score, gs.next_player())){
        best_score = val.score;
        best_moves.clear();
        best_moves.push_back(m);
        switch (gs.next_player()){
        case Player::Black:
          if (best_score > alpha)
            alpha = best_score;
          break;
        case Player::White:
          if (best_score < beta)
            beta = best_score;
          break;
        default: assert(false);
        }
      } else if (val.score == best_score)
        best_moves.push_back(m);
    }
    if (best_moves.size() > 0){
      uint choice = s::rand() % best_moves.size();
      return RRet(best_score, best_moves[choice]);
    } else
      return RRet(best_score, Move(M::Pass));
  }
public:
  MinimaxAgent(size_t depth, EvalFn& eval): mDepth(depth), mEval(eval) {
    s::srand(unsigned(s::time(0)));
  }

  Move select_move(const GameState& gs){
    if constexpr (ABP){
      RRet ret = recursive_alpha_beta_minimax_search(gs, mDepth, MIN_SCORE, MAX_SCORE);
      return ret.move;
    } else {
      RRet ret = recursive_minimax_search(gs, mDepth);
      return ret.move;
    }
  }
};

} // rlgames

#endif//RLGAMES_MINIMAX_AGENT
