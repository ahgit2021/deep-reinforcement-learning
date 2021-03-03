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
    env_info = env.reset(train_mode=False)[brain_name]
    brain = env.brains[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    agent = Agent(state_size, action_size, random_seed=0, num_steps_update=0, num_updates=0)

    for i in range(2):
        agent.actor_local[i].load_state_dict(torch.load('checkpoint_actor' + str(i) + '.pth'))

    episode_scores = []

    for episode in range(num_episodes):
        scores = np.zeros(num_agents) # initialize the score (for each agent)

        while True:
            states = env_info.vector_observations
            actions = agent.act(states, add_noise=False) # select an action (for each agent)
            env_info = env.step(actions)[brain_name] # send all actions to tne environment
            scores += env_info.rewards

            if np.any(env_info.local_done): # exit loop if episode finished
                break

        episode_score = np.max(scores)
        episode_scores.append(episode_score)
        running_mean = np.mean(episode_scores[-100:])

        print('Episode {}\tScore: {:.2f}\tLast 10 Scores: {:.2f}\tRunning Average: {:.2f}\n'.format(episode, episode_score, np.mean(episode_scores[-10:]), running_mean), end="")

    return episode_scores
