import numpy as np

class ZeroExperienceCollector:
  def __init__(self):
    self.states = []
    self.vcounts = []
    self.rewards = []
    self.curr_eps_states = []
    self.curr_eps_vcounts = []

  def begin_episode(self):
    self.curr_eps_states = []
    self.curr_eps_vcounts = []

  def record_decision(self, state, visit_counts):
    self.curr_eps_states.append(state)
    self.curr_eps_vcounts.append(visit_counts)

  def complete_episode(self, reward):
    num_states = len(self.curr_eps_states)
    self.states += self.curr_eps_states
    self.vcounts += self.curr_eps_vcounts
    self.rewards += [reward for _ in range(num_states)]
    self.curr_eps_states = []
    self.curr_eps_vcounts = []

  def clear(self):
    self.states = []
    self.vcounts =  []
    self.rewards = []

class ZeroExperienceBuffer:
  def __init__(self, states, visit_counts, rewards):
    self.states = states
    self.vcounts = visit_counts
    self.rewards = rewards

  def serialize(self, h5file):
    h5file.create_group('experience')
    h5file['experience'].create_dataset('states', data=self.states)
    h5file['experience'].create_dataset('visit_counts', data=self.vcounts)
    h5file['experience'].create_dataset('rewards', data=self.rewards)

def combine_experience(collectors):
  combined_states = np.concatenate([np.array(c.states) for c in collectors])
  combined_vcounts = np.concatenate([np.array(c.vcounts) for c in collectors])
  combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
  return ZeroExperienceBuffer(combined_states, combined_vcounts, combined_rewards)

def load_experience(h5file):
  return ExperienceBuffer(
      np.array(h5file['experience']['states']),
      np.array(h5file['experience']['visit_counts']),
      np.array(h5file['experience']['rewards']))
