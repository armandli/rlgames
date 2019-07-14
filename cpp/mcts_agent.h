#ifndef RLGAMES_MCTS_AGENT
#define RLGAMES_MCTS_AGENT

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <set>
#include <vector>
#include <algorithm>

#include <type_alias.h>
#include <types.h>
#include <bag.h>
#include <agent_base.h>

namespace s = std;

namespace rlgames {

//TODO: make MCTS more optimized
//TODO: find faster random number generator method
//TODO: find faster method than keep doing modulo for random number generation
//TODO: monitor and avoid object copying for GameState, MCTSNode

template <typename Board, typename GameState>
struct MCTSAgent : AgentBase<Board, GameState, MCTSAgent<Board, GameState>> {
  static constexpr float MIN_SCORE = -10.F;
  static constexpr float MAX_SCORE =  10.F;
  static constexpr float TIE_SCORE =  0.F;
private:
  size_t mMaxExpand;
  float  mEFactor;
  size_t mMCBatchSize;
  s::vector<udyte> mCache;
protected:
  void initialize_cache(size_t sz){
    mCache.resize(sz);
    s::iota(mCache.begin(), mCache.end(), 0);
  }

  float mc_play(GameState& gs){
    IsPointAnEye<Board> is_point_an_eye;
    float score = TIE_SCORE;
    for (size_t i = 0; i < mMCBatchSize; ++i){
      GameState play_state = gs;
      while (not play_state.is_over()){
        s::random_shuffle(s::begin(mCache), s::end(mCache), [](int k){ return s::rand() % k; });
        for (udyte index : mCache){
          Pt pt = point<Board::SIZE>(index);
          Move m(M::Play, pt);
          if (play_state.is_valid_move(m) && (not is_point_an_eye(play_state.board(), pt, play_state.next_player()))){
            play_state.apply_move(m);
            break;
          }
        }
      }
      switch (play_state.winner()){
      case Player::Black:   score += MAX_SCORE; break;
      case Player::White:   score += MIN_SCORE; break;
      case Player::Unknown: score += TIE_SCORE; break;
      default: assert(false);
      }
    }
    return score;
  }

  bool is_improvement(float score, float best_score, Player player){
    switch (player){
    case Player::Black: return score > best_score;
    case Player::White: return score < best_score;
    default: assert(false);
    }
  }

  float init_best_score(Player player){
    switch (player){
    case Player::Black: return MIN_SCORE;
    case Player::White: return MAX_SCORE;
    default: assert(false);
    }
  }

  struct MCTSNode {
    GameState            gs;
    MCTSNode*            parent;
    float                qvalue;
    size_t               ncount;
    bag<Move>            unp_children; //unexplored children, initially all legal moves
    s::vector<MCTSNode*> children;     //all expanded children
    //TODO: maybe a better container, DenseSet ?
    s::set<MCTSNode*>    ptn_children; //all expanded children that has unexplored children

    MCTSNode() = default;
    explicit MCTSNode(GameState& gs, MCTSNode* parent = nullptr):
      gs(gs), parent(parent), qvalue(0.), ncount(0), unp_children(gs.legal_moves()) {
    }
    MCTSNode(const MCTSNode& o):
      gs(o.gs), parent(o.parent), qvalue(o.qvalue), ncount(o.ncount), unp_children(o.unp_children), children(o.children), ptn_children(o.ptn_children) {}
    MCTSNode& operator=(const MCTSNode& o){
      gs           = o.gs;
      parent       = o.parent;
      qvalue       = o.qvalue;
      ncount       = o.ncount;
      unp_children = o.unp_children;
      children     = o.children;
      ptn_children = o.ptn_children;
      return *this;
    }

    //TODO: reformulate MCTSNode methods to make it easier to use
    void add_child(MCTSNode* node, size_t index){
      assert(node != nullptr);
      unp_children.pop(index);
      children.push_back(node);
      ptn_children.insert(node);
    }
    void remove_child(MCTSNode* node){
      assert(node != nullptr);
      ptn_children.erase(node);
      if (parent && unp_children.size() == 0 && ptn_children.size() == 0)
        parent->remove_child(this);
    }
    void update(float qv, size_t cnt){
      qvalue += qv;
      ncount += cnt;
      if (parent)
        parent->update(qv, cnt);
    }
  };

  template <typename T>
  struct BufferAllocator {
    size_t       index;
    s::vector<T> objects;

    explicit BufferAllocator(size_t sz): index(0U) {
      objects.resize(sz);
    }
    T* allocate(const T& obj){
      assert(index < objects.size());

      objects[index] = obj;
      size_t prev_index = index;
      index++;
      return &objects[prev_index];
    }
  };

  //TODO: make sure the recursion does not build up stack
  MCTSNode* recursive_uct(MCTSNode* node, BufferAllocator<MCTSNode>& arena){
    if (node->unp_children.size() > 0){
      //TODO: why so complicated here?
      uint rand_idx = s::rand() % node->unp_children.size();
      Move new_move = node->unp_children[rand_idx];
      GameState new_state = node->gs;
      new_state.apply_move(new_move);
      MCTSNode* new_node = arena.allocate(MCTSNode(new_state, node));
      node->add_child(new_node, rand_idx);
      if (new_state.is_over())
        node->remove_child(new_node);
      return new_node;
    } else {
      s::vector<MCTSNode*> best_children;
      float best_score = init_best_score(node->gs.next_player());
      for (decltype(s::begin(node->ptn_children)) it = s::begin(node->ptn_children); it != s::end(node->ptn_children); ++it){
        float score = (*it)->qvalue / (float)(*it)->ncount;
        float explore_factor = mEFactor * s::sqrt(2 * s::log((float)node->ncount / (float)(*it)->ncount));
        if ((*it)->gs.next_player() == Player::Black)
          score += explore_factor;
        else
          score -= explore_factor;
        if (is_improvement(score, best_score, node->gs.next_player())){
          best_score = score;
          best_children.clear();
          best_children.push_back(*it);
        } else if (score == best_score)
          best_children.push_back(*it);
      }
      if (best_children.size() > 0){
        uint random_choice = s::rand() % best_children.size();
        return recursive_uct(best_children[random_choice], arena);
      } else
        return nullptr;
    }
  }
public:
  MCTSAgent(size_t max_expansion, float exploration_factor, size_t mc_sample_size = 1):
    mMaxExpand(max_expansion), mEFactor(exploration_factor), mMCBatchSize(mc_sample_size) {
    s::srand(unsigned(s::time(0)));
  }

  Move select_move(GameState& gs){
    if (mCache.size() == 0){
      size_t sz = gs.board().size();
      initialize_cache(sz * sz);
    }
    BufferAllocator<MCTSNode> arena(mMaxExpand);
    MCTSNode* root = arena.allocate(MCTSNode(gs));

    assert(root != nullptr);

    for (size_t i = 0; i < mMaxExpand; ++i){
      MCTSNode* node = recursive_uct(root, arena);
      if (node == nullptr) break;

      float qvalue = mc_play(node->gs);
      node->update(qvalue, mMCBatchSize);
    }
    float best_score = init_best_score(root->gs.next_player());
    s::vector<MCTSNode*> best_nodes;
    for (MCTSNode* child : root->children){
      float score = child->qvalue / (float)child->ncount;
      if (is_improvement(score, best_score, root->gs.next_player())){
        best_score = score;
        best_nodes.clear();
        best_nodes.push_back(child);
      } else if (score == best_score)
        best_nodes.push_back(child);
    }
    if (best_nodes.size() > 0){
      uint choice = s::rand() % best_nodes.size();
      return best_nodes[choice]->gs.previous_move();
    } else
      return Move(M::Pass);
  }
};

} // rlgames

#endif//RLGAMES_MCTS_AGENT
