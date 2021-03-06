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

    oracle = Agent(state_size, action_size, random_seed=0, num_steps_update=20, num_updates=10)

    for i in range(2):
        oracle.actor_local[i].load_state_dict(torch.load('checkpoint_actor_oracle' + str(i) + '.pth'))
    oracle.critic_local.load_state_dict(torch.load('checkpoint_critic_oracle.pth'))

    agent.oracle = oracle

    episode_scores = []
    solved = False

    for episode in range(num_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
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
            scores += rewards                                  # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            curstep += 1
            if np.any(dones):                                  # exit loop if episode finished
                break

        episode_score = np.max(scores)
        episode_scores.append(episode_score)
        running_mean = np.mean(episode_scores[-100:])

        print('Episode {}\tScore: {:.2f}\tLast 10 Scores: {:.2f}\tRunning Average: {:.2f}\tSteps: {}\n'.format(episode, episode_score, np.mean(episode_scores[-10:]), running_mean, curstep), end="")
        if running_mean >= .5 and len(episode_scores) >= 100:
            print("solved in {} episodes!\n".format(episode + 1))
            for i in range(2):
                torch.save(agent.actor_local[i].state_dict(), 'checkpoint_actor' + str(i) + '.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            solved = True
            break
        agent.update()

    return solved, episode_scores
