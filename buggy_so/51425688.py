import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

def get_random_action(epsilon):
    return np.random.rand(1) < epsilon

def get_reward_prediction(q, a):
    qs_a = np.concatenate((q, table[a]), axis=0)
    x = np.zeros(shape=(1, environment_parameters + num_of_possible_actions))
    x[0] = qs_a
    guess = model.predict(x[0].reshape(1, x.shape[1]))
    r = guess[0][0]
    return r

results = []
epsilon = 0.05 #@DRLinter-->epsilon_decay
alpha = 0.003 #@DRLinter-->alpha
gamma = 0.3 #@DRLinter-->gamma
environment_parameters = 8
num_of_possible_actions = 4
obs = 15
mem_max = 100000
epochs = 3 #@DRLinter-->epoch_count
total_episodes = 15000

possible_actions = np.arange(0, num_of_possible_actions)
table = np.zeros((num_of_possible_actions, num_of_possible_actions))
table[np.arange(num_of_possible_actions), possible_actions] = 1

env = gym.make('LunarLander-v2') #@DRLinter-->create_env
env.reset() #@DRLinter-->initialize_env_correct

i_x = np.random.random((5, environment_parameters + num_of_possible_actions))
i_y = np.random.random((5, 1))

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=i_x.shape[1]))
model.add(Dense(i_y.shape[1]))

opt = optimizers.adam(lr=alpha)

model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

total_steps = 0
i_x = np.zeros(shape=(1, environment_parameters + num_of_possible_actions))
i_y = np.zeros(shape=(1, 1))

mem_x = np.zeros(shape=(1, environment_parameters + num_of_possible_actions))
mem_y = np.zeros(shape=(1, 1))
max_steps = 40000

for episode in range(total_episodes):
    g_x = np.zeros(shape=(1, environment_parameters + num_of_possible_actions))
    g_y = np.zeros(shape=(1, 1))
    q_t = env.reset()
    episode_reward = 0

    for step_number in range(max_steps):
        if episode < obs:
            a = env.action_space.sample()
        else:
            if get_random_action(epsilon, total_episodes, episode):
                a = env.action_space.sample() #@DRLinter-->exploration_check
            else:
                actions = np.zeros(shape=num_of_possible_actions)

                for i in range(4):
                    actions[i] = get_reward_prediction(q_t, i)

                a = np.argmax(actions)

        # env.render()
        qa = np.concatenate((q_t, table[a]), axis=0)

        s, r, episode_complete, data = env.step(a) #@DRLinter-->in_correct_step
        episode_reward += r

        if step_number is 0:
            g_x[0] = qa
            g_y[0] = np.array([r])
            mem_x[0] = qa
            mem_y[0] = np.array([r])

        g_x = np.vstack((g_x, qa))
        g_y = np.vstack((g_y, np.array([r])))

        if episode_complete:
            for i in range(0, g_y.shape[0]):
                if i is 0:
                    g_y[(g_y.shape[0] - 1) - i][0] = g_y[(g_y.shape[0] - 1) - i][0]
                else:
                    g_y[(g_y.shape[0] - 1) - i][0] = g_y[(g_y.shape[0] - 1) - i][0] + gamma * g_y[(g_y.shape[0] - 1) - i + 1][0] #update_equation

            if mem_x.shape[0] is 1:
                mem_x = g_x
                mem_y = g_y
            else:
                mem_x = np.concatenate((mem_x, g_x), axis=0)
                mem_y = np.concatenate((mem_y, g_y), axis=0)

            if np.alen(mem_x) >= mem_max:
                for l in range(np.alen(g_x)):
                    mem_x = np.delete(mem_x, 0, axis=0)
                    mem_y = np.delete(mem_y, 0, axis=0)

        q_t = s

        if episode_complete and episode >= obs:
            if episode%10 == 0:
                model.fit(mem_x, mem_y, batch_size=32, epochs=epochs, verbose=0)

        if episode_complete:
            results.append(episode_reward)
            break #@DRLinter-->terminal_state
