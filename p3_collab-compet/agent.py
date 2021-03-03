import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim
import sys

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NOISE_DECAY=0.9995

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, num_steps_update, num_updates):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.num_steps_update = num_steps_update
        self.num_updates = num_updates
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.noise_scale = 1.0

        # Actor Network (w/ Target Network)
        self.actor_local = [Actor(state_size, action_size, random_seed).to(device) for i in range(2)]
        self.actor_target = [Actor(state_size, action_size, random_seed).to(device) for i in range(2)]
        self.actor_optimizer = [optim.Adam(self.actor_local[i].parameters(), lr=LR_ACTOR) for i in range(2)]

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * 2, action_size * 2, random_seed).to(device)
        self.critic_target = Critic(state_size * 2, action_size * 2, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done, curstep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        assert state.shape == (2, self.state_size)
        assert next_state.shape == (2, self.state_size)
        assert action.shape == (2, self.action_size)
        assert len(done) == 2 and done[0] == done[1]
        assert len(reward) == 2
        self.memory.add(state, action, sum(reward), next_state, done[0])

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and curstep % self.num_steps_update == 0:
            for i in range(self.num_updates):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        action = []
        for i in range(2):
            self.actor_local[i].eval()
            with torch.no_grad():
                action.append(self.actor_local[i](state[i]).cpu().data.numpy())
            self.actor_local[i].train()
            if add_noise:
                action[i] += np.random.normal(scale=self.noise_scale, size=action[i].shape)
        return np.clip(action, -1, 1)

    def update(self):
        self.noise_scale *= NOISE_DECAY

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, done = experiences
        batch_size = BATCH_SIZE
        assert states.shape == (batch_size, 2, self.state_size)
        assert next_states.shape == (batch_size, 2, self.state_size)
        assert actions.shape == (batch_size, 2, self.action_size)
        assert rewards.shape == (batch_size,)
        assert done.shape == (batch_size,)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = torch.cat([self.actor_target[i](next_states.permute(1,0,2)[i]) for i in range(2)], dim=1)
        assert actions_next.shape== (batch_size, self.action_size * 2)
        Q_targets_next = self.critic_target(next_states.reshape(batch_size, -1), actions_next).squeeze()
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - done))
        # Compute critic loss
        Q_expected = self.critic_local(states.reshape(batch_size, -1), actions.reshape(batch_size, -1)).squeeze()
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = torch.cat([self.actor_local[i](states.permute(1,0,2)[i]) for i in range(2)], dim=1)
        assert actions_pred.shape== (batch_size, self.action_size * 2)
        actor_loss = -self.critic_local(states.reshape(batch_size, -1), actions_pred).mean()
        # Minimize the loss
        for i in range(2):
            self.actor_optimizer[i].zero_grad()
        actor_loss.backward()
        for i in range(2):
            self.actor_optimizer[i].step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        for i in range(2):
            self.soft_update(self.actor_local[i], self.actor_target[i], TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.Tensor([e.state for e in experiences if e is not None]).float().to(device)
        actions = torch.Tensor([e.action for e in experiences if e is not None]).float().to(device)
        rewards = torch.Tensor([e.reward for e in experiences if e is not None]).float().to(device)
        next_states = torch.Tensor([e.next_state for e in experiences if e is not None]).float().to(device)
        dones = torch.Tensor([np.uint8(e.done) for e in experiences if e is not None]).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
