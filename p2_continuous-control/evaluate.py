import numpy as np
import random
import copy
from collections import namedtuple, deque
import model
import agent

import importlib
importlib.reload(model)
importlib.reload(agent)

from model import Actor, Critic
from agent import Agent

import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
import collections

sys.dont_write_bytecode = True

def evaluate_agent(env, num_episodes):
  brain_name = env.brain_names[0]
  brain = env.brains[brain_name]
  env_info = env.reset(train_mode=False)[brain_name]
  num_agents = len(env_info.agents)
  action_size = brain.vector_action_space_size
  states = env_info.vector_observations
  state_size = states.shape[1]

  agent = Agent(state_size, action_size, random_seed = 0, num_steps_update = 0, num_updates = 0)
  agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
  mean_scores = []

  for episode in range(num_episodes):
    scores = np.zeros(num_agents)
    while True:
      actions = agent.act(states, add_noise=False) # select an action (for each agent)
      env_info = env.step(actions)[brain_name]           # send all actions to tne environment
      next_states = env_info.vector_observations         # get next state (for each agent)
      rewards = env_info.rewards                         # get reward (for each agent)
      dones = env_info.local_done                        # see if episode finished
      scores += env_info.rewards                         # update the score (for each agent)
      states = next_states                               # roll over states to next time step
      if np.any(dones):                                  # exit loop if episode finished
        break

    mean_scores.append(np.mean(scores))
    print('Episode {}\tScore: {:.2f}\tRunning Average: {:.2f}'.format(episode, mean_scores[-1], np.mean(mean_scores)))

  print('Mean score over 100 episodes: {:.2f}'.format(np.mean(mean_scores)))
  return mean_scores
