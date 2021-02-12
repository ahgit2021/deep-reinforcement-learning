import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from agent import Agent

import torch
import torch.nn.functional as F
import torch.optim as optim

def train_agent(env, num_episodes):
  brain_name = env.brain_names[0]
  env_info = env.reset(train_mode=True)[brain_name]
  brain = env.brains[brain_name]
  num_agents = len(env_info.agents)
  action_size = brain.vector_action_space_size
  states = env_info.vector_observations
  state_size = states.shape[1]
  agent = Agent(state_size, action_size, random_seed=0)

  episode = 0

  while episode < num_episodes:
    states = env_info.vector_observations
    state_size = states.shape[1]
    scores = np.zeros(num_agents) # initialize the score (for each agent)
    steps = 0
    while True:
      actions = agent.act(states) # select an action (for each agent)
      env_info = env.step(actions)[brain_name]           # send all actions to tne environment
      next_states = env_info.vector_observations         # get next state (for each agent)
      rewards = env_info.rewards                         # get reward (for each agent)
      dones = env_info.local_done                        # see if episode finished
      agent.step(states, actions, rewards, next_states, dones)
      scores += env_info.rewards                         # update the score (for each agent)
      states = next_states                               # roll over states to next time step
      steps += 1
      if np.any(dones):                                  # exit loop if episode finished
        print('Total score (averaged over agents) episode {}: {}'.format(episode, np.mean(scores)))
        break
    episode += 1 
