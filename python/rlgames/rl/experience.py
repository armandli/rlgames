import numpy as np

class ExperienceBuffer:
  def __init__(self, states, actions, rewards):
    self.states  = states
    self.actions = actions
    self.rewards = rewards

  def serialize(self, h5file):
    h5file.create_group('experience')
    h5file['experience'].create_dataset('states', data=self.states)
    h5file['experience'].create_dataset('actions', data=self.actions)
    h5file['experience'].create_dataset('rewards', data=self.rewards)

class AdvExperienceBuffer:
  def __init__(self, states, actions, rewards, advantages):
    self.states     = states
    self.actions    = actions
    self.rewards    = rewards
    self.advantages = advantages

  def serialize(self, h5file):
    h5file.create_group('experience')
    h5file['experience'].create_dataset('states', data=self.states)
    h5file['experience'].create_dataset('actions', data=self.actions)
    h5file['experience'].create_dataset('rewards', data=self.rewards)
    h5file['experience'].create_dataset('advantages', data=self.advantages)

def load_experience(h5file):
  return ExperienceBuffer(
    states  = np.array(h5file['experience']['states']),
    actions = np.array(h5file['experience']['actions']),
    rewards = np.array(h5file['experience']['rewards']))

def load_adv_experience(h5file):
  return AdvExperienceBuffer(
    states  = np.array(h5file['experience']['states']),
    actions = np.array(h5file['experience']['actions']),
    rewards = np.array(h5file['experience']['rewards']),
    advantages = np.array([h5file['experience']['advantages']]))

class ExperienceCollector:
  def __init__(self):
    self.states       = []
    self.actions      = []
    self.rewards      = []
    self.curr_states  = []
    self.curr_actions = []

  def begin_episode(self):
    self.curr_states  = []
    self.curr_actions = []

  def record_decision(self, state, action):
    self.curr_states.append(state)
    self.curr_actions.append(action)

  def complete_episode(self, reward):
    num_states = len(self.curr_states)
    self.states  += self.curr_states
    self.actions += self.curr_actions
    self.rewards += [reward for _ in range(num_states)] #no reward decay
    self.curr_states  = []
    self.curr_actions = []

  def to_buffer(self):
    return ExperienceBuffer(
      states  = np.array(self.states),
      actions = np.array(self.actions),
      rewards = np.array(self.rewards))
  
  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.curr_states = []
    self.curr_actions = []

class AdvExperienceCollector:
  def __init__(self):
    self.states       = []
    self.actions      = []
    self.rewards      = []
    self.advantages   = []
    self.curr_states  = []
    self.curr_actions = []
    self.curr_est_v   = [] #estimate of V(s)

  def begin_episode(self):
    self.curr_states  = []
    self.curr_actions = []
    self.curr_est_v   = []

  def record_decision(self, state, action, estimated_value):
    self.curr_states.append(state)
    self.curr_actions.append(action)
    self.curr_est_v.append(estimated_value)

  def complete_episode(self, reward):
    num_states = len(self.curr_states)
    self.states  += self.curr_states
    self.actions += self.curr_actions
    self.rewards += [reward for _ in range(num_states)] #no reward decay
    for i in range(num_states):
      advantage = reward - self.curr_est_v[i]
      self.advantages.append(advantage)
    self.curr_states  = []
    self.curr_actions = []
    self.curr_est_v   = []

  def to_buffer(self):
    return ExperienceBuffer(
      states     = np.array(self.states),
      actions    = np.array(self.actions),
      rewards    = np.array(self.rewards),
      advantages = np.array(self.advantages))

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.advantages = []
    self.curr_states = []
    self.curr_actions = []
    self.curr_est_v = []

def combine_experience(collectors):
  combined_states  = np.concatenate([np.array(c.states) for c in collectors])
  combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
  combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
  return ExperienceBuffer(
    states=combined_states,
    actions=combined_actions,
    rewards=combined_rewards)

def adv_combine_experience(collectors):
  combined_states     = np.concatenate([np.array(c.states) for c in collectors])
  combined_actions    = np.concatenate([np.array(c.actions) for c in collectors])
  combined_rewards    = np.concatenate([np.array(c.rewards) for c in collectors])
  combined_advantages = np.concatenate([np.array(c.advantages) for c in collectors])
  return AdvExperienceBuffer(
    states=combined_states,
    actions=combined_actions,
    rewards=combined_rewards,
    advantages=combined_advantages)

