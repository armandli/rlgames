import numpy as np

#simple add-it-up game to demonstrate credit assignment and gradual improvement of policy favoring winning policy over losing policy

choices = [1,2,3,4,5]
learning_rate = 0.0001

def simulate_game(policy):
  p1_choices = {1:0,2:0,3:0,4:0,5:0}
  p2_choices = {1:0,2:0,3:0,4:0,5:0}
  p1_total = 0
  p2_total = 0
  for i in range(100):
    p1_choice = np.random.choice(choices, p=policy)
    p1_choices[p1_choice] += 1
    p2_total += p1_choice
    p2_choice = np.random.choice(choices, p=policy)
    p2_choices[p2_choice] += 1
    p2_total += p2_choice
  if p1_total > p2_total:
    winner_choices = p1_choices
    loser_choices = p2_choices
  else:
    winner_choices = p2_choices
    loser_choices = p1_choices
  return (winner_choices, loser_choices)

def normalize_policy(policy):
  policy = np.clip(policy, 0., 1.)
  return policy / np.sum(policy)

def main():
  policy = [0.2,0.2,0.2,0.2,0.2]
  num_games = 1000000
  for i in range(num_games):
    win_counts, lose_counts = simulate_game(policy)
    for j, choice in enumerate(choices):
      net_wins = win_counts[choice] - lose_counts[choice]
      policy[j] += learning_rate * net_wins
    policy = normalize_policy(policy)
    print('%d: %s' % (i, policy))

if __name__ == '__main__':
  main()
