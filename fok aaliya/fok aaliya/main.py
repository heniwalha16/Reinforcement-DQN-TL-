import numpy as np
from SumoEnvironment import SumoEnvironment
from dqn_agent import DQNAgent

def train_dqn_agent(env, agent, episodes, batch_size):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])

        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}")

if __name__ == "__main__":
    sumo_cfg_file = "haya.sumocfg"  # Path to your SUMO configuration file
    env = SumoEnvironment(sumo_cfg_file)

    state_size = len(env.get_state())  # Assuming get_state() returns the full state
    action_size = 2  # Define your action size (e.g., 2 for green and red light control)

    agent = DQNAgent(state_size, action_size)
    episodes = 100  # Number of training episodes
    batch_size = 32  # Batch size for experience replay

    train_dqn_agent(env, agent, episodes, batch_size)

    # Optionally save the trained model
    # agent.save("dqn_traffic_model.h5")
