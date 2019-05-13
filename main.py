import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import os
from tqdm import tqdm
import datetime
from agent import PPOAgent

from unityagents import UnityEnvironment
import numpy as np


def test_agent(env, brain_name, agent, device, real_time=False):
    env_info = env.reset(train_mode=not real_time)[brain_name]
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)
    while True:
        states = torch.FloatTensor(states).to(device)
        action, _, _ = agent.act(states)
        env_info = env.step(action.cpu().data.numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += rewards
        states = next_states
        if np.any(dones):
            break
    return np.mean(scores)


def plot(scores=[], ylabel="Scores", xlabel="Episode #", title="", text=""):
    fig, ax = plt.subplots()

    for score in scores:
        ax.plot(np.arange(len(score)), score)
    xlabel = "\n".join([xlabel, text])
    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=title)
    ax.grid()
    #    fig.text(-.2,-.2,text)
    fig.tight_layout()
    fig.savefig(f"plot_{datetime.datetime.now().isoformat().replace(':', '')}.png")
    plt.show()

def main():
    os.environ['NO_PROXY'] = 'localhost,127.0.0.*'

    # Hyper params:
    hidden_size = 256
    lr = 3e-4
    num_steps = 2048
    mini_batch_size = 512
    ppo_epochs = 3
    threshold_reward = 10
    max_episodes = 15  # 1e5
    episode = 0
    nrmlz_adv = True
    test_mean_reward = 1.


    scores = [

#    run_experiment(hidden_size=256, lr=1e-3, max_episodes=30, mini_batch_size=512,
#                                                   nrmlz_adv=False, num_steps=2048, ppo_epochs=4, threshold_reward=20),


    run_experiment(hidden_size=256, lr=1e-3, max_episodes=200, mini_batch_size=128,
                                                      nrmlz_adv=True, num_steps=2048, ppo_epochs=4, threshold_reward=30, clip_gradients=True),

    run_experiment(hidden_size=256, lr=1e-3, max_episodes=30, mini_batch_size=32,
                                                      nrmlz_adv=True, num_steps=2048, ppo_epochs=4, threshold_reward=20, clip_gradients=True),

    run_experiment(hidden_size=256, lr=1e-3, max_episodes=30, mini_batch_size=128,
                                                      nrmlz_adv=True, num_steps=2048, ppo_epochs=4, threshold_reward=20, clip_gradients=False)
    ]
    plot([x[0] for x in scores], "Scores")


def run_experiment(hidden_size, lr, max_episodes, mini_batch_size, nrmlz_adv, num_steps, ppo_epochs, threshold_reward, clip_gradients):
    scores_window, test_rewards = experiment(hidden_size=hidden_size, lr=lr, num_steps=num_steps,
                                             mini_batch_size=mini_batch_size, ppo_epochs=ppo_epochs,
                                             threshold_reward=threshold_reward, max_episodes=max_episodes,
                                             nrmlz_adv=nrmlz_adv, clip_gradients=clip_gradients)


    test_mean_reward = np.mean(test_rewards)
    text = "\n".join([f"HS:{hidden_size} lr:{lr} st:{num_steps} batch:{mini_batch_size} ppo:{ppo_epochs}",
                      f" r:{threshold_reward} e:{max_episodes} adv:{nrmlz_adv} mean {test_mean_reward}"])
    plot([scores_window], "Last # Scores", text=text)
    return scores_window, test_rewards


def experiment(hidden_size=64, lr=3e-4, num_steps=2048, mini_batch_size=32, ppo_epochs=10, threshold_reward=10,
               max_episodes=15, nrmlz_adv=True, clip_gradients=True):
    '''

    :param hidden_size: number of neurons for the layers of the model
    :param lr: learning rate
    :param num_steps: maximum duration of one epoch
    :param mini_batch_size: mini batch size for ppo
    :param ppo_epochs: number of epochs for ppo to learn
    :param threshold_reward: what is the goal of the training
    :param max_episodes: maximum duration of the training
    :param nrmlz_adv: True, if advantages should be normalized before PPO
    :param clip_gradients: True if gradients should ne clipped after PPO
    :return: list of scores and list of test_rewards
    '''

    use_cuda = torch.cuda.is_available()
    #    device   = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    print(device)
    scores_window = deque(maxlen=100)

    test_rewards = []

    env = UnityEnvironment(file_name='reacher20/reacher', base_port=64739)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = PPOAgent(learning_rate=lr, state_size=state_size, action_size=action_size, hidden_size=hidden_size,
                     num_agents=num_agents, random_seed=0, ppo_epochs=ppo_epochs,
                     mini_batch_size=mini_batch_size, normalize_advantages=nrmlz_adv, clip_gradients= clip_gradients, device=device)


    #    while episode < max_episodes and not early_stop:
    for episode in tqdm(range(max_episodes)):
        log_probs = []
        values = []
        states_list = []
        actions_list = []
        rewards = []
        masks = []
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        for duration in range(num_steps):

            state = torch.FloatTensor(state).to(device)
            action, value, log_prob = agent.act(state)
            env_info = env.step(action.cpu().data.numpy())[brain_name]  # send all actions to the environment

            next_state = env_info.vector_observations  # get next state (for each agent)
            reward = env_info.rewards  # get reward (for each agent)
            dones = np.array(env_info.local_done)  # see if episode finished
            if reward == None:
                pass

            log_probs.append(log_prob)
            values.append(value)
            reward_t = torch.FloatTensor(reward).unsqueeze(1).to(device)
            masks_t = torch.FloatTensor(1 - dones)
            rewards.append(reward_t)
            masks.append(masks_t)
            states_list.append(state)
            actions_list.append(action)

            state = next_state

            if np.any(dones):
                break

        next_state = torch.FloatTensor(state).to(device)
        _, next_value, _ = agent.act(next_state)
        agent.step(states=states_list, actions=actions_list, values=values,
                   log_probs=log_probs, rewards=rewards, masks=masks, next_value=next_value)

        test_mean_reward = test_agent(env, brain_name, agent, device)
        test_rewards.append(test_mean_reward)
        scores_window.append(test_mean_reward)
        print('Episode {}, Total score this episode: {}, Last {} average: {}'.format(episode, test_mean_reward,
                                                                                     min(episode, 100),
                                                                                     np.mean(scores_window)))
        if np.mean(scores_window) > threshold_reward:
            agent.save_model(f"ppo_checkpoint_{test_mean_reward}_e{episode}_hs{hidden_size}_lr{lr}_st{num_steps}_b{mini_batch_size}_ppo{ppo_epochs}_r{threshold_reward}_e{episode}_adv{nrmlz_adv}_{test_mean_reward}.pth")
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, test_mean_reward))
            break

        episode += 1
    env.close()
    return scores_window, test_rewards

if __name__ == "__main__":
    main()
