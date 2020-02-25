#ifndef RLGAMES_ZERO_AGENT
#define RLGAMES_ZERO_AGENT

#include <type_alias.h>
#include <dirichlet_distribution.h>
#include <models/model_base.h>
#include <encoders/go_action_encoder.h>
#include <encoders/go_zero_encoder.h>
#include <experience/zero_episodic_buffer.h>
#include <agents/agent_base.h>

#include <cassert>
#include <array>
#include <algorithm>

#include <torch/torch.h>

namespace rlgames {

namespace s = std;
namespace t = torch;

template <typename Model, typename NoiseDist, typename RGen, typename Board, typename GameState, uint BF>
class ZeroAgent : public AgentBase<Board, GameState, ZeroAgent<Model, NoiseDist, RGen, Board, GameState, BF>> {
public:
  static constexpr float MIN_SCORE = -1.F;
  static constexpr float MAX_SCORE =  1.F;
  static constexpr float TIE_SCORE =  0.F;
private:
  Model&                    mModel;
  NoiseDist                 mNoise;
  t::Device                 mDevice;
  RGen                      mGen;
  ZeroEpisodicExpCollector* mExp;
  uint                      mMaxExpand;
  float                     mEFactor;
  float                     mNoiseFactor;
protected:
  struct Node;

  //keeps tracks the statistics of each move from some node
  struct Branch {
    float prior;
    float total_value;
    long  visit_count;
    Node* child;
    explicit Branch(float prior = 0.F, float total_value = 0, uint visit_count = 0):
      prior(prior),
      total_value(total_value),
      visit_count(visit_count),
      child(nullptr)
    {}
  };

  struct Node {
    GameState gs;
    float     qvalue; //TODO: this is actually never used after node creation!
    uint      total_count;
    Branch    branches[BF];
    Node*     parent;
    uint      last_midx; //array index of the last move, not Move

    Node(GameState&& gs, float qvalue, float* priors, Node* parent = nullptr, uint last_midx = BF):
      gs(s::move(gs)),
      qvalue(qvalue),
      total_count(1U),
      parent(parent),
      last_midx(last_midx){
      for (uint i = 0; i < BF; ++i)
        branches[i].prior = priors[i];
    }
    Node& operator=(Node&& o){
      gs = s::move(o.gs);
      qvalue = o.qvalue;
      total_count = o.total_count;
      s::memcpy(branches, o.branches, sizeof(Branch) * BF);
      parent = o.parent;
      last_midx = o.last_midx;
      return *this;
    }
    Node() = default;

    void add_child(uint midx, Node* child_node){
      branches[midx].child = child_node;
    }
    bool has_child(uint midx){
      if (midx >= BF) return false;
      else            return branches[midx].child != nullptr;
    }
    Node* child(uint midx){
      return branches[midx].child;
    }
    float expected_value(uint midx){
      return (branches[midx].visit_count > 0) ?
        branches[midx].total_value / (float)branches[midx].visit_count :
        0.F;
    }
    float prior(uint midx){
      return branches[midx].prior;
    }
    uint visit_count(uint midx){
      return branches[midx].visit_count;
    }
    void record_visit(uint midx, float value){
      total_count += 1;
      branches[midx].visit_count += 1;
      branches[midx].total_value += value;
    }
  };

  template <typename T>
  struct BufferAllocator {
    size_t index;
    s::vector<T> objects;

    explicit BufferAllocator(size_t sz): index(0U), objects(sz){}

    T* allocate(T&& obj) noexcept {
      assert(index < objects.size());

      objects[index] = s::move(obj);
      size_t prev_index = index;
      index++;
      return &objects[prev_index];
    }
  };

  Node* create_node(BufferAllocator<Node>& arena, GameState&& gs, Node* parent = nullptr, uint midx = BF){
    TensorP state = mModel.state_encoder.encode_state(gs, mDevice);
    TensorP avout = mModel.model->forward(state);
    t::Tensor priors = avout.x.to(t::Device(t::kCPU));
    Node* new_node = arena.allocate(Node(s::move(gs), avout.y.item().to<float>(), (float*)priors.data_ptr(), parent, midx));
    if (parent != nullptr){
      assert(midx != BF);
      parent->add_child(midx, new_node);
    }
    return new_node;
  }

  uint select_branch(Node* node){
    assert(node != nullptr);
    if (node->gs.is_over()) return BF;

    s::vector<Move> moves = node->gs.legal_moves(); //TODO: cannot switch to relaxed_legal_moves, due to violating 0 liberty rule
    s::array<float, BF> noise = mNoise(mGen);
    float tcount = node->total_count;
    s::array<float, BF> score;
    s::fill(s::begin(score), s::end(score), MIN_SCORE);
    for (uint i = 0; i < moves.size(); ++i){
      uint idx = mModel.action_encoder.move_to_idx(moves[i]);
      float q = node->expected_value(idx);
      float p = node->prior(idx);
      float n = node->visit_count(idx);
      score[idx] = q + mEFactor * ((1 - mNoiseFactor) * p + mNoiseFactor * noise[idx]) * (s::sqrt(tcount) / (n + 1));
    }

    uint max_idx = random_max_element(s::begin(score), s::end(score), [](float a, float b){return a < b;}) - s::begin(score);
    return max_idx;
  }

  void append_experience(Node& root){
    if (mExp){
      TensorP state = mModel.state_encoder.encode_state(root.gs, mDevice);
      s::vector<float> visit_counts(BF);
      for (uint i = 0; i < BF; ++i)
        visit_counts[i] = root.branches[i].visit_count;
      mExp->append(state, visit_counts);
    }
  }

  template <typename Iter, typename Less>
  Iter random_max_element(Iter begin, Iter end, Less less){
    s::vector<Iter> max_iters;
    Iter it = begin;
    max_iters.push_back(it);
    it++;
    while (it != end){
      if (not less(*it, *max_iters.front())){
        if (less(*max_iters.front(), *it))
          max_iters.clear();
        max_iters.push_back(it);
      }
      it++;
    }
    return max_iters[mGen() % max_iters.size()];
  }
public:
  ZeroAgent(Model& model, t::Device device, uint max_expansion, float exploration_factor, float noise_alpha, float noise_factor, uint seed):
    mModel(model),
    mNoise(noise_alpha),
    mDevice(device),
    mGen(seed),
    mExp(nullptr),
    mMaxExpand(max_expansion),
    mEFactor(exploration_factor),
    mNoiseFactor(noise_factor)
  {}

  Move select_move(const GameState& gs){
    //root should never be a terminal state
    assert(not gs.is_over());

    BufferAllocator<Node> arena(mMaxExpand + 1);
    GameState gs_copy = gs;
    Node* root = create_node(arena, s::move(gs_copy));
    for (uint r = 0; r < mMaxExpand; ++r){
      Node* node = root;
      uint next_midx = select_branch(node);
      while (node->has_child(next_midx)){
        node = node->child(next_midx);   //node can be nullptr
        next_midx = select_branch(node); //terminal state has no next_midx
      }
      float value;
      if (next_midx >= BF){
        //we reached terminal state, update visit count, we cannot choose
        //to explore other nodes because visit count indicate best choice
        value = -1.F * node->qvalue;
        next_midx = node->last_midx;
        node = node->parent;
      } else {
        //we have not expanded this node
        GameState new_gs = node->gs;
        Move move = mModel.action_encoder.idx_to_move(next_midx);
        new_gs.apply_move(move);
        Node* new_node = create_node(arena, s::move(new_gs), node, next_midx);
        value = -1.F * new_node->qvalue;
      }
      // backup the tree to update visit counts
      while (node != nullptr){
        node->record_visit(next_midx, value);
        next_midx = node->last_midx;
        node = node->parent;
        value = -1.F * value;
      }
    }
    //collects experience, for AlphaZero, it's the visit count
    //to select a move, pick the immediate branch with the highest visit
    //count
    append_experience(*root);

    int max_midx = random_max_element(s::begin(root->branches), s::end(root->branches), [](Branch& a, Branch& b){ return a.visit_count < b.visit_count; }) - s::begin(root->branches);
    return mModel.action_encoder.idx_to_move(max_midx);
  }

  void set_exp(ZeroEpisodicExpCollector& exp){
    mExp = &exp;
  }
};

} // rlgames

#endif//RLGAMES_ZERO_AGENT
