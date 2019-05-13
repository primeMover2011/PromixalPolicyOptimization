import torch
import torch.optim as optim
import torch.nn as nn
from network import PPOActorCritic

from IPython.display import clear_output
import matplotlib.pyplot as plt
from collections import deque
import os
from tqdm import tqdm
import datetime

from unityagents import UnityEnvironment
import numpy as np

class PPOAgent():

    def __init__(self, learning_rate, state_size, action_size, hidden_size,
                 num_agents, random_seed, ppo_epochs, mini_batch_size, normalize_advantages, clip_gradients, device):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random_seed
        self.learning_rate=learning_rate#
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.nrmlz_adv = normalize_advantages
        self.clip_gradients = clip_gradients

        self.ppo_model = PPOActorCritic(num_inputs = state_size,
                                        num_outputs = action_size,
                                        hidden_size=hidden_size).to(device)
        self.optimizer = optim.Adam(self.ppo_model.parameters(), lr=learning_rate, eps=1e-5)

        pass

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
    #    dist, value = self.ppo_model(state)
        dist, value = self.ppo_model(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        #log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return action, value, log_prob

    def step(self, states, actions, values, log_probs, rewards, masks, next_value):

        returns = self.compute_gaes(next_value, rewards, masks, values)

        returns = torch.cat(returns).detach()
        mean2 = torch.mean(returns)
        print("Returns: ", mean2)
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()

        states = torch.cat(states)
        actions = torch.cat(actions)
        advantages = returns - values
        if self.nrmlz_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        losses = []
        clip_param = 0.2
        self.learn(states=states, actions=actions, log_probs=log_probs,
                   returns=returns, advantages=advantages, clip_param=clip_param)



    def save_model(self, file_name):
        torch.save(self.ppo_modelppomodel.state_dict(), file_name)

    def load_model(self, file_name):
        pass


    def sample(self, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], \
                  log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]


    def learn(self, states, actions, log_probs, returns, advantages,
                   clip_param=0.2):

        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.sample(states, actions, log_probs,
                                                                             returns, advantages):
                dist, value = self.ppo_model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action).sum(dim=1, keepdim=True)

                ratio = (new_log_probs - old_log_probs).exp()
                # surrogate objective
                surr1 = ratio * advantage
                # clipped surrogate objective
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_gradients:
                    nn.utils.clip_grad_norm_(self.ppo_model.parameters(), 5)

                self.optimizer.step()


    def compute_gaes(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        advantage = 0
        returns = []
        for step in reversed(range(len(rewards))):
            td_error = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            advantage = advantage * tau * gamma * masks[step] + td_error
            returns.insert(0, advantage + values[step])
        return returns
