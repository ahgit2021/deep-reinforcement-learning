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

def train_agent(env, num_episodes):
  brain_name = env.brain_names[0]
  env_info = env.reset(train_mode=True)[brain_name]
  brain = env.brains[brain_name]
  num_agents = len(env_info.agents)
  action_size = brain.vector_action_space_size
  states = env_info.vector_observations
  state_size = states.shape[1]
  agent = Agent(state_size, action_size, random_seed=0, num_steps_update=20, num_updates=10)

  episode = 0
  mean_scores = []
  running_mean_scores = collections.deque(maxlen=100)

  while episode < num_episodes:
    states = env_info.vector_observations
    state_size = states.shape[1]
    scores = np.zeros(num_agents) # initialize the score (for each agent)
    curstep = 0
    while True:
      actions = agent.act(states) # select an action (for each agent)
      env_info = env.step(actions)[brain_name]           # send all actions to tne environment
      next_states = env_info.vector_observations         # get next state (for each agent)
      rewards = env_info.rewards                         # get reward (for each agent)
      dones = env_info.local_done                        # see if episode finished
      agent.step(states, actions, rewards, next_states, dones, curstep)
      scores += env_info.rewards                         # update the score (for each agent)
      states = next_states                               # roll over states to next time step
      curstep += 1
      if np.any(dones):                                  # exit loop if episode finished
        break

    mean_score_episode = np.mean(scores)
    mean_scores.append(mean_score_episode)
    running_mean_scores.append(mean_score_episode)
    running_mean = np.mean(running_mean_scores)

    print('Episode {}\tScore: {:.2f}\tLast 10 Scores: {:.2f}\tRunning Average: {:.2f}\n'.format(episode, np.mean(mean_scores[-10:]), mean_score_episode, running_mean), end="")
    if running_mean >= 30:
      print("solved in {} episodes!\n".format(episode))
      torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
      torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
      break
    episode += 1
    agent.update()
