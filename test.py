import numpy as np
import torch
from agent import PPOAgent
from unityagents import UnityEnvironment



def test_agent(env, brain_name, agent, device, real_time=False):
    env_info = env.reset(train_mode=not real_time)[brain_name]
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)
    while True:
        states = torch.FloatTensor(states).to(device)
        action, _, _ = agent.act(states, train=True)
        env_info = env.step(action.cpu().data.numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += rewards
        states = next_states
        if np.any(dones):
            break
    return np.mean(scores)

def main():

    device   = torch.device("cpu")
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

    agent = PPOAgent(state_size=state_size, action_size=action_size, hidden_size=256,
                     num_agents=num_agents, random_seed=0, ppo_epochs=4,
                     mini_batch_size=128, normalize_advantages=True, learning_rate=3e-4,
                     clip_gradients=True, gamma=0.99, tau=0.95, device=device)
    agent.load_model('assets/ppo_checkpoint_37.10.pth')
    test_agent(env, brain_name, agent, device, real_time=True)


if __name__ == "__main__":
    main()
